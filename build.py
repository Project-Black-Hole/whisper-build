#!/usr/bin/env python3
"""Builds whisper.cpp's server, with one patch, from pinned sources.

    python build.py pin --whisper <tag> [--vulkan-sdk <version>|latest]
        Downloads the sources, hashes them and writes sources.pin.json.
    python build.py build [--variant vulkan|cpu] [--tag <tag>]
        Checks every source against its SHA-256, applies
        patches/whisper-server.patch, builds `whisper-server` with ggml's
        backends as loadable libraries (every CPU variant; Vulkan for the
        `vulkan` variant), gathers the program and the libraries it needs,
        checks what they depend on, starts the built server and checks that
        it answers only its token and that a request's times are its own,
        and writes dist/whisper-runtime-<tag>-<variant>-x64.zip.
    python build.py smoke --dir <folder> --model <ggml model>
                          --vad-model <ggml model> --wav <file>
        Runs the same checks against a server that is already built. The
        model must be a real one (the source's `for-tests` models say
        nothing, and the check of the times needs words).

Standard library only. `--cache <folder>` (pin and build) keeps the
downloaded sources there and uses them when their hashes match.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIN = ROOT / "sources.pin.json"
PATCH = ROOT / "patches" / "whisper-server.patch"
WORK = ROOT / "work"
DIST = ROOT / "dist"
WINDOWS = os.name == "nt"
EXE = "whisper-server.exe" if WINDOWS else "whisper-server"
TOKEN_VARIABLE = "PBH_WHISPER_TOKEN"

# The Visual C++ runtime's libraries: never taken from the system folder of
# the machine that builds, always from the compiler's own redistributable
# folder, and put beside the program.
VC_RUNTIME = re.compile(r"^(vcruntime\d+(_\d+)?|msvcp\d+(_\w+)?|vcomp\d+|concrt\d+|vccorlib\d+)\.dll$", re.IGNORECASE)
# Parts of Windows itself that are not files in its system folder.
API_SET = re.compile(r"^(api-ms-win-|ext-ms-)", re.IGNORECASE)
# The Vulkan loader: a GPU driver installs it. Only the Vulkan backend may
# need it, so a computer without one still runs the server on its CPU.
VULKAN_LOADER = "vulkan-1.dll"

CMAKE_OPTIONS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DGGML_NATIVE=OFF",
    "-DWHISPER_BUILD_TESTS=OFF",
    "-DWHISPER_BUILD_EXAMPLES=ON",
    "-DWHISPER_BUILD_SERVER=ON",
    "-DWHISPER_SDL2=OFF",
    "-DWHISPER_CURL=OFF",
]


def say(text=""):
    print(text, flush=True)


def fail(text):
    say(f"FAIL  {text}")
    sys.exit(1)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "whisper-build"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            partial.replace(dest)
            return
        except (urllib.error.URLError, OSError) as error:
            say(f"      attempt {attempt} failed: {error}")
            time.sleep(5 * attempt)
    fail(f"could not download {url}")


def read_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": "whisper-build"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", "replace").strip()
    except (urllib.error.URLError, OSError):
        return None


def entry_for(dest):
    return {"file": dest.name, "sha256": sha256(dest), "bytes": dest.stat().st_size}


def vulkan_sdk_urls(version):
    """The installer's address has had more than one name."""
    base = f"https://sdk.lunarg.com/sdk/download/{version}/windows"
    return [
        f"{base}/vulkansdk-windows-X64-{version}.exe",
        f"{base}/VulkanSDK-{version}-Installer.exe",
        f"{base}/vulkan_sdk.exe",
    ]


def cmd_pin(args):
    cache = Path(args.cache)
    tag = args.whisper
    version = tag.lstrip("v")
    pin = json.loads(PIN.read_text(encoding="utf-8")) if PIN.exists() else {}

    dest = cache / f"whisper.cpp-{version}.tar.gz"
    url = f"https://github.com/ggml-org/whisper.cpp/archive/refs/tags/{tag}.tar.gz"
    say(f"whisper.cpp {tag}: {url}")
    download(url, dest)
    pin["whisper"] = {"version": version, "tag": tag, "url": url, **entry_for(dest)}
    say(f"      {pin['whisper']['sha256']}  {pin['whisper']['bytes']} bytes")

    if args.vulkan_sdk:
        wanted = args.vulkan_sdk
        if wanted == "latest":
            wanted = read_text("https://vulkan.lunarg.com/sdk/latest/windows.txt")
            if not wanted or not re.fullmatch(r"[0-9.]+", wanted):
                listed = read_text("https://vulkan.lunarg.com/sdk/latest/windows.json") or ""
                found = re.search(r'"windows"\s*:\s*"([0-9.]+)"', listed)
                wanted = found.group(1) if found else None
            if not wanted:
                fail("the newest Vulkan SDK version could not be read; give --vulkan-sdk <version>")
            say(f"the newest Vulkan SDK: {wanted}")
        dest = cache / f"vulkan-sdk-{wanted}-windows-x64.exe"
        found = None
        for url in vulkan_sdk_urls(wanted):
            say(f"Vulkan SDK {wanted}: {url}")
            partial = dest.with_name(dest.name + ".try")
            request = urllib.request.Request(url, headers={"User-Agent": "whisper-build"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
                    shutil.copyfileobj(response, out, 1 << 20)
            except (urllib.error.URLError, OSError) as error:
                say(f"      not there ({error})")
                continue
            with open(partial, "rb") as handle:
                head = handle.read(2)
            if head != b"MZ" or partial.stat().st_size < 20_000_000:
                say("      not an installer")
                continue
            partial.replace(dest)
            found = url
            break
        if not found:
            fail(f"no installer of the Vulkan SDK {wanted} was found")
        pin["vulkanSdk"] = {"version": wanted, "url": found, **entry_for(dest)}
        say(f"      {pin['vulkanSdk']['sha256']}  {pin['vulkanSdk']['bytes']} bytes")
        published = read_text(f"https://sdk.lunarg.com/sdk/sha/{wanted}/windows/{found.rsplit('/', 1)[1]}.txt")
        if published:
            same = published.split()[0].lower() == pin["vulkanSdk"]["sha256"]
            say(f"      LunarG's own SHA-256 for it: {'the same' if same else 'DIFFERENT: ' + published}")
            if not same:
                fail("the download does not match the SHA-256 LunarG publishes")
        else:
            say("      LunarG's own SHA-256 for it could not be read (not checked against it)")
    PIN.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8", newline="\n")
    say(f"wrote {PIN.name}")


def obtain(entry, cache):
    """A pinned source in the cache, downloaded if it is not there, and
    refused unless its SHA-256 is the pinned one."""
    dest = Path(cache) / entry["file"]
    if not (dest.exists() and sha256(dest) == entry["sha256"]):
        say(f"      downloading {entry['url']}")
        download(entry["url"], dest)
    actual = sha256(dest)
    if actual != entry["sha256"] or dest.stat().st_size != entry["bytes"]:
        fail(f"{dest.name} is not the pinned file (SHA-256 {actual}, {dest.stat().st_size} bytes)")
    say(f"PASS  {dest.name} matches its pin")
    return dest


def extract(archive, dest):
    """Unpacks a .tar.gz whose entries all sit under one folder; returns it.
    An entry that would land outside `dest` stops the build."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    root = dest.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (dest / member.name).resolve()
            if root != target and root not in target.parents:
                fail(f"{archive.name}: {member.name} would be written outside the folder")
            if member.issym() or member.islnk():
                link = (target.parent / member.linkname).resolve()
                if root != link and root not in link.parents:
                    fail(f"{archive.name}: {member.name} links outside the folder")
        tar.extractall(dest)
    tops = [p for p in dest.iterdir()]
    if len(tops) != 1 or not tops[0].is_dir():
        fail(f"{archive.name} does not hold one folder")
    return tops[0]


def run(command, env=None, cwd=None):
    say("      " + " ".join(str(part) for part in command))
    result = subprocess.run([str(part) for part in command], env=env, cwd=cwd)
    if result.returncode != 0:
        fail(f"{Path(str(command[0])).name} ended with {result.returncode}")


def apply_patch(source):
    """The patch must apply whole and exactly: `git apply` refuses a hunk
    that does not match, with no fuzz."""
    run(["git", "apply", "--check", "--verbose", PATCH], cwd=source)
    run(["git", "apply", "--verbose", PATCH], cwd=source)
    patched = (source / "examples" / "server" / "server.cpp").read_text(encoding="utf-8")
    for must in (TOKEN_VARIABLE, "set_pre_routing_handler", "[pbh] listening on"):
        if must not in patched:
            fail(f"the patched server.cpp does not hold {must!r}")
    if 'request_path + "/load"' in patched or "Access-Control-Allow-Origin" in patched:
        fail("the patched server.cpp still has /load or a cross-origin header")
    # Upstream clears the table once (inside whisper_vad); the patch adds a
    # place for each of the two calls that can run without voice detection.
    library = (source / "src" / "whisper.cpp").read_text(encoding="utf-8")
    cleared = library.count("state->vad_mapping_table.clear();")
    if library.count("[pbh]") != 2 or cleared < 4:
        fail(f"the patched whisper.cpp does not clear the voice detection's time table in both calls ({cleared} place(s))")
    say("PASS  the patch applied: token, Origin refusal, no /load, no cross-origin headers, no time table kept from an earlier request")


def install_vulkan_sdk(entry, cache):
    """Installs the pinned Vulkan SDK into work/ (headers, the loader's
    import library and the shader compiler are what the build needs)."""
    installer = obtain(entry, cache)
    root = WORK / "vulkan-sdk"
    shutil.rmtree(root, ignore_errors=True)
    run([installer, "--root", root, "--accept-licenses", "--default-answer", "--confirm-command", "install"])
    # Where it was asked to go, or where the installer puts it by itself.
    places = [root, Path(os.environ.get("SystemDrive", "C:") + "\\") / "VulkanSDK" / entry["version"]]
    for place in places:
        glslc = place / "Bin" / "glslc.exe"
        if glslc.exists() and (place / "Include" / "vulkan" / "vulkan.h").exists():
            say(f"PASS  Vulkan SDK {entry['version']} installed ({place})")
            return place
    fail(f"the Vulkan SDK has no Bin\\glslc.exe and Include\\vulkan\\vulkan.h after its install (looked in: {', '.join(str(p) for p in places)})")


# ---- What a Windows program or library needs (its import table) ----

def imports_pe(path):
    """The names of the libraries a PE file imports (load-time and
    delay-loaded), lower case."""
    data = Path(path).read_bytes()
    if data[:2] != b"MZ":
        fail(f"{path} is not a Windows program")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        fail(f"{path} has no PE header")
    sections = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic != 0x20B:
        fail(f"{path} is not a 64-bit program")
    directories = optional + 112
    table = optional + optional_size
    layout = []
    for index in range(sections):
        at = table + index * 40
        virtual_size, virtual_address, raw_size, raw_at = struct.unpack_from("<IIII", data, at + 8)
        layout.append((virtual_address, max(virtual_size, raw_size), raw_at))

    def offset(rva):
        for virtual_address, size, raw_at in layout:
            if virtual_address <= rva < virtual_address + size:
                return rva - virtual_address + raw_at
        fail(f"{path}: address {rva:#x} is in no section")

    def text(rva):
        start = offset(rva)
        end = data.index(b"\0", start)
        return data[start:end].decode("ascii", "replace").lower()

    names = []
    import_rva = struct.unpack_from("<I", data, directories + 1 * 8)[0]
    if import_rva:
        at = offset(import_rva)
        while True:
            name_rva = struct.unpack_from("<I", data, at + 12)[0]
            if name_rva == 0:
                break
            names.append(text(name_rva))
            at += 20
    delay_rva = struct.unpack_from("<I", data, directories + 13 * 8)[0]
    if delay_rva:
        at = offset(delay_rva)
        while True:
            name_rva = struct.unpack_from("<I", data, at + 4)[0]
            if name_rva == 0:
                break
            names.append(text(name_rva))
            at += 32
    return sorted(set(names))


def redist_library(name):
    """A Visual C++ runtime library from the newest redistributable folder
    of the Visual Studio that is installed (never from System32)."""
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        fail(f"vswhere.exe is not at {vswhere}")
    out = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                         capture_output=True, text=True)
    install = Path(out.stdout.strip().splitlines()[0]) if out.stdout.strip() else None
    if not install or not install.exists():
        fail("Visual Studio's folder could not be found (vswhere)")
    found = []
    for folder in (install / "VC" / "Redist" / "MSVC").glob("*/x64/Microsoft.VC*"):
        if "debug" in str(folder).lower():
            continue
        candidate = folder / name
        if candidate.exists():
            version = tuple(int(n) for n in re.findall(r"\d+", folder.parent.parent.name))
            found.append((version, candidate))
    if not found:
        fail(f"{name} is in no redistributable folder under {install}")
    return sorted(found)[-1][1]


def gather_windows(bin_dir, stage, variant):
    """Copies the server and what it needs into `stage`; returns what came
    from where. Every import is accounted for: a file of the build, a
    Visual C++ runtime library from the redistributable folder, a part of
    Windows, or (for the Vulkan backend alone) the Vulkan loader."""
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    built = {p.name.lower(): p for p in bin_dir.iterdir() if p.suffix.lower() in (".dll", ".exe")}
    if EXE not in built:
        fail(f"{EXE} is not in {bin_dir}")
    wanted = [EXE] + sorted(n for n in built if re.fullmatch(r"ggml-[\w.\-]+\.dll", n))
    if variant == "vulkan" and "ggml-vulkan.dll" not in wanted:
        fail("the build made no ggml-vulkan.dll")
    if variant != "vulkan" and "ggml-vulkan.dll" in wanted:
        fail("a build without Vulkan made a ggml-vulkan.dll")
    if not any(n.startswith("ggml-cpu-") for n in wanted):
        fail("the build made no ggml-cpu-*.dll")
    report = {"built": [], "runtime": [], "system": [], "vulkanLoader": []}
    seen = set()
    queue = list(wanted)
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        if name in built:
            source = built[name]
            report["built"].append(name)
        elif VC_RUNTIME.match(name):
            source = redist_library(name)
            report["runtime"].append(name)
        else:
            fail(f"{name} is neither built here nor a Visual C++ runtime library")
        shutil.copy2(source, stage / source.name)
        for needed in imports_pe(source):
            if needed in seen or needed in built or VC_RUNTIME.match(needed):
                queue.append(needed)
            elif API_SET.match(needed):
                continue
            elif needed == VULKAN_LOADER:
                if name != "ggml-vulkan.dll":
                    fail(f"{name} needs the Vulkan loader: only ggml-vulkan.dll may, or a computer without Vulkan can't start the server")
                report["vulkanLoader"].append(name)
            elif (system32 / needed).exists():
                if needed not in report["system"]:
                    report["system"].append(needed)
            else:
                fail(f"{name} needs {needed}, which is not built here, not a runtime library and not part of Windows")
    for key in report:
        report[key] = sorted(report[key])
    say(f"PASS  gathered {len(report['built'])} built file(s) and {len(report['runtime'])} runtime librar(ies)")
    say(f"      built: {', '.join(report['built'])}")
    say(f"      Visual C++ runtime, beside the program: {', '.join(report['runtime']) or 'none'}")
    say(f"      parts of Windows it uses: {', '.join(report['system'])}")
    if variant == "vulkan" and report["vulkanLoader"] != ["ggml-vulkan.dll"]:
        fail("ggml-vulkan.dll does not import the Vulkan loader: is it a Vulkan build?")
    return report


def gather_unix(bin_dir, stage):
    """Linux: the program and every library of the build, flat. Not a
    release (nothing is published for Linux); it lets the checks below run
    where the build was written."""
    names = []
    for path in sorted(bin_dir.iterdir()):
        if path.name == EXE or re.search(r"\.so(\.\d+)*$", path.name):
            if re.match(r"lib(parakeet)", path.name):
                continue
            shutil.copy2(path.resolve(), stage / path.name)
            names.append(path.name)
    if EXE not in names:
        fail(f"{EXE} is not in {bin_dir}")
    return {"built": names, "runtime": [], "system": [], "vulkanLoader": []}


# ---- The built server, asked ----

def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def ask(port, path, token=None, origin=None, form=None, method=None):
    """One request; returns (status, headers, body). Status 0: the server
    closed the connection before an answer could be read whole. That is what
    a refusal can look like from outside when an upload was on its way: the
    server answers before it reads a request's body, and closing a
    connection with unread data resets it (seen on Windows)."""
    headers = {}
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if origin:
        headers["Origin"] = origin
    if form is not None:
        boundary = "pbhbuild" + hashlib.sha256(os.urandom(16)).hexdigest()[:24]
        parts = []
        for name, value in form.items():
            if isinstance(value, bytes):
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="part.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + value + b"\r\n")
            else:
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        data = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=300) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        try:
            body = error.read()
        except OSError:
            body = b""
        return error.code, dict(error.headers), body
    except (urllib.error.URLError, OSError) as error:
        return 0, {}, str(error).encode("utf-8", "replace")


def with_silence_before(sound, seconds):
    """A WAV file's sound with `seconds` of silence put before it."""
    with wave.open(io.BytesIO(sound)) as source:
        shape = source.getparams()
        frames = source.readframes(shape.nframes)
    out = io.BytesIO()
    with wave.open(out, "wb") as dest:
        dest.setparams(shape)
        dest.writeframes(b"\0" * (shape.framerate * shape.sampwidth * shape.nchannels * seconds) + frames)
    return out.getvalue()


def times_of(answer):
    """A verbose_json answer's segments as (start, end), in seconds."""
    try:
        return [(float(s["start"]), float(s["end"])) for s in answer.get("segments", [])]
    except (AttributeError, KeyError, TypeError, ValueError):
        return []


def smoke(folder, model, wav, vad_model):
    """Starts the server in `folder` and checks the patch from outside."""
    exe = Path(folder) / EXE
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("GGML_", "WHISPER_")) and k != TOKEN_VARIABLE}
    if not WINDOWS:
        env["LD_LIBRARY_PATH"] = str(folder)
    log = WORK / "smoke-stderr.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    problems = []

    def check(label, ok, detail=""):
        say(f"{'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}")
        if not ok:
            problems.append(label)

    # Without a token it does not start at all.
    port = free_port()
    arguments = [str(exe), "--host", "127.0.0.1", "--port", str(port), "--model", str(model),
                 "--vad-model", str(vad_model)]
    bare = subprocess.run(arguments, env=env, cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    check("no token: the server refuses to start (exit code 4)", bare.returncode == 4, f"exit code {bare.returncode}")
    short = subprocess.run(arguments, env={**env, TOKEN_VARIABLE: "short"}, cwd=folder,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    check("a token under 32 characters: the same", short.returncode == 4, f"exit code {short.returncode}")

    token = hashlib.sha256(os.urandom(32)).hexdigest()
    with open(log, "wb") as err:
        server = subprocess.Popen(arguments, env={**env, TOKEN_VARIABLE: token}, cwd=folder,
                                  stdout=subprocess.DEVNULL, stderr=err)
    try:
        line = f"[pbh] listening on 127.0.0.1:{port}"
        deadline = time.time() + 180
        up = False
        while time.time() < deadline and server.poll() is None:
            if line in log.read_text(encoding="utf-8", errors="replace"):
                up = True
                break
            time.sleep(0.2)
        check("the server says on standard error that it holds the port", up)
        if not up:
            say(log.read_text(encoding="utf-8", errors="replace")[-3000:])
            return problems
        sound = Path(wav).read_bytes()
        status, headers, body = ask(port, "/health", token=token)
        check("the token: /health answers", status == 200 and b'"ok"' in body, f"HTTP {status}")
        cross = [h for h in headers if h.lower().startswith("access-control")]
        check("no cross-origin header in an answer", not cross, ", ".join(cross))
        for label, kwargs in [
            ("no token", {}),
            ("another token", {"token": token[:-1] + ("0" if token[-1] != "0" else "1")}),
        ]:
            status, _, body = ask(port, "/health", **kwargs)
            check(f"{label}: /health is refused with 401", status == 401, f"HTTP {status}")
            status, _, _ = ask(port, "/inference", form={"response_format": "json"}, **kwargs)
            check(f"{label}: /inference is refused with 401", status == 401, f"HTTP {status}")
            # With a recording on its way the refusal comes before the upload
            # is read: the 401, or the connection closed under the upload.
            status, _, _ = ask(port, "/inference", form={"file": sound, "response_format": "json"}, **kwargs)
            check(f"{label}: an upload to /inference is not taken (401, or the connection is closed)",
                  status in (401, 0), "the connection was closed" if status == 0 else f"HTTP {status}")
            check(f"{label}: the server is still running after it", server.poll() is None)
            status, _, _ = ask(port, "/", **kwargs)
            check(f"{label}: the server's own page is refused with 401", status == 401, f"HTTP {status}")
        status, _, _ = ask(port, "/health", token=token, origin=f"http://127.0.0.1:{port}")
        check("the token and an Origin header: refused with 403", status == 403, f"HTTP {status}")
        status, headers, _ = ask(port, "/inference", origin="https://example.com", method="OPTIONS")
        cross = [h for h in headers if h.lower().startswith("access-control")]
        check("a browser's preflight: refused with 403 and no cross-origin header", status == 403 and not cross, f"HTTP {status}")
        status, _, _ = ask(port, "/load", token=token, form={"model": str(exe)})
        check("the token: /load does not exist (404)", status == 404, f"HTTP {status}")
        check("the server is still running after /load", server.poll() is None)
        status, _, body = ask(port, "/inference", token=token,
                              form={"file": sound, "response_format": "verbose_json", "no_language_probabilities": "true"})
        answer = {}
        try:
            answer = json.loads(body)
        except ValueError:
            pass
        check("the token: a recording is transcribed (verbose_json)",
              status == 200 and "segments" in answer and answer.get("duration", 0) > 0, f"HTTP {status}")

        # A request's times are its own. Upstream keeps the time table of the
        # last request that had voice detection on and maps the next request
        # without it through that table. Asked here: a recording with 20
        # seconds of silence before it and voice detection on (its table
        # moves every time by about 20 seconds), then the recording itself
        # with voice detection off, on the same server.
        def heard(label, data, vad):
            status, _, body = ask(port, "/inference", token=token,
                                  form={"file": data, "response_format": "verbose_json",
                                        "no_language_probabilities": "true", "vad": vad})
            try:
                got = json.loads(body)
            except ValueError:
                got = {}
            if status != 200 or not isinstance(got, dict):
                got = {}
            times = times_of(got)
            say(f"      {label}: HTTP {status}, {float(got.get('duration') or 0):.2f} s, segments "
                + (", ".join(f"{a:.2f}-{b:.2f}" for a, b in times) or "none"))
            return float(got.get("duration") or 0), times

        lead = 20
        try:
            padded = with_silence_before(sound, lead)
        except (wave.Error, EOFError) as error:
            padded = None
            check("the recording is a WAV file silence can be put before", False, str(error))
        if padded is not None:
            length, before = heard("voice detection off", sound, "false")
            check("voice detection off: the model says words (a real model is needed for the checks of the times)",
                  length > 0 and len(before) > 0, f"{len(before)} segment(s)")
            _, shifted = heard(f"{lead} s of silence first, voice detection on", padded, "true")
            check(f"voice detection on: the first segment starts after the silence ({lead - 5} s or later)",
                  len(shifted) > 0 and shifted[0][0] >= lead - 5,
                  f"it starts at {shifted[0][0]:.2f} s" if shifted else "no segment")
            length, after = heard("voice detection off again", sound, "false")
            inside = len(after) > 0 and all(a < length and b <= length + 1.0 for a, b in after)
            check("then voice detection off on the same server: every segment lies inside its own recording",
                  inside, (f"the last one starts at {after[-1][0]:.2f} s of {length:.2f} s" if after else "no segment"))
            say(f"      the same request before and after: {'the same times' if before == after else 'DIFFERENT times'}")

        status, _, _ = ask(port, "/inference", token=token, form={"file": b"not a recording", "response_format": "json"})
        check("the token: a file that is no recording is refused with 400", status == 400, f"HTTP {status}")
        status, _, _ = ask(port, "/health", token=token)
        check("the server still answers at the end", status == 200, f"HTTP {status}")
        text = log.read_text(encoding="utf-8", errors="replace")
        check("the token is not in the server's log", token not in text)
        for wanted in ("whisper_backend_init_gpu:", "load_backend:"):
            hits = [l for l in text.splitlines() if l.startswith(wanted)]
            for hit in hits[:12]:
                say(f"      {hit}")
    finally:
        server.kill()
        server.wait()
    return problems


def cmd_smoke(args):
    problems = smoke(Path(args.dir).resolve(), Path(args.model).resolve(), Path(args.wav).resolve(),
                     Path(args.vad_model).resolve())
    if problems:
        fail(f"{len(problems)} check(s) failed")
    say("SMOKE: PASS")


# ---- Licences ----

MIT_TEXT = """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def write_licenses(source, dest, runtime):
    """The licence of everything that is in the package, from the sources
    themselves: a missing notice stops the build."""
    dest.mkdir(parents=True)
    shutil.copy2(source / "LICENSE", dest / "whisper.cpp-and-ggml-LICENSE.txt")

    httplib = (source / "examples" / "server" / "httplib.h").read_text(encoding="utf-8", errors="replace")[:600]
    owner = re.search(r"Copyright \(c\) (\d{4}(?:-\d{4})?) ([^.\n]+)\.", httplib)
    if not owner or "MIT License" not in httplib:
        fail("httplib.h no longer names its copyright and the MIT License at its top")
    (dest / "cpp-httplib-LICENSE.txt").write_text(
        f"cpp-httplib (examples/server/httplib.h)\n\nMIT License\n\nCopyright (c) {owner.group(1)} {owner.group(2)}\n\n{MIT_TEXT}",
        encoding="utf-8", newline="\n")

    nlohmann = (source / "examples" / "json.hpp").read_text(encoding="utf-8", errors="replace")[:1200]
    owner = re.search(r"SPDX-FileCopyrightText: (.+)", nlohmann)
    if not owner or "SPDX-License-Identifier: MIT" not in nlohmann:
        fail("json.hpp no longer names its copyright and the MIT licence at its top")
    (dest / "nlohmann-json-LICENSE.txt").write_text(
        f"JSON for Modern C++ (examples/json.hpp)\n\nMIT License\n\nCopyright (c) {owner.group(1).strip()}\n\n{MIT_TEXT}",
        encoding="utf-8", newline="\n")

    for name, out in (("miniaudio.h", "miniaudio-LICENSE.txt"), ("stb_vorbis.c", "stb_vorbis-LICENSE.txt")):
        text = (source / "examples" / name).read_text(encoding="utf-8", errors="replace")
        at = text.rfind("This software is available")
        if at < 0 or "Public Domain" not in text[at:]:
            fail(f"{name} no longer ends with its licence")
        (dest / out).write_text(f"{name} (examples/{name}); its own words:\n\n{text[at:].rstrip()}\n",
                                encoding="utf-8", newline="\n")

    notes = [
        "What is in this package, and under which licence",
        "",
        "whisper-server, whisper and the ggml libraries: whisper.cpp and ggml, MIT",
        "  (whisper.cpp-and-ggml-LICENSE.txt), built from the pinned source with the",
        "  patch in this package's build repository (patches/whisper-server.patch),",
        "  which is offered under the same MIT licence as the file it changes.",
        "Inside whisper-server: cpp-httplib, MIT (cpp-httplib-LICENSE.txt); JSON for",
        "  Modern C++, MIT (nlohmann-json-LICENSE.txt); miniaudio, public domain or",
        "  MIT No Attribution (miniaudio-LICENSE.txt); stb_vorbis, MIT or public",
        "  domain (stb_vorbis-LICENSE.txt).",
    ]
    if runtime:
        notes += [
            f"{', '.join(runtime)}: the Microsoft Visual C++ runtime, copied",
            "  unchanged from Visual Studio's redistributable folder and distributed",
            "  under Microsoft's terms for redistributable code.",
        ]
    notes += [
        "The Vulkan shaders inside ggml-vulkan are whisper.cpp's own, compiled by the",
        "  Vulkan SDK's glslc at build time. Nothing of the SDK is in this package;",
        "  the Vulkan loader (vulkan-1.dll) is the computer's own.",
        "",
    ]
    (dest / "README.txt").write_text("\n".join(notes), encoding="utf-8", newline="\n")


def write_zip(stage, dest):
    """Every file of `stage` at the top of the archive, in name order, with
    one fixed time, so the same files give the same archive."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(stage).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.name == EXE else 0o644) << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def cmd_build(args):
    started = time.time()
    if platform.machine().lower() not in ("x86_64", "amd64"):
        fail(f"only x64 is built here, this is {platform.machine()}")
    variant = args.variant
    if variant == "vulkan" and not WINDOWS:
        fail("the Vulkan variant is built on Windows; use --variant cpu here")
    tag = args.tag or os.environ.get("GITHUB_REF_NAME") or "dev"
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.\-]*", tag):
        fail(f"{tag!r} can't be a release's name")
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    cache = Path(args.cache) if args.cache else WORK / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    if tag != "dev" and not tag.startswith(pin["whisper"]["version"] + "-"):
        fail(f"the tag {tag} is not <whisper.cpp version>-<build number> for the pinned {pin['whisper']['version']}")

    say("== sources")
    archive = obtain(pin["whisper"], cache)
    source = extract(archive, WORK / "src")
    apply_patch(source)

    env = dict(os.environ)
    options = list(CMAKE_OPTIONS)
    if variant == "vulkan":
        say("== Vulkan SDK")
        sdk = install_vulkan_sdk(pin["vulkanSdk"], cache)
        env["VULKAN_SDK"] = str(sdk)
        env["PATH"] = str(sdk / "Bin") + os.pathsep + env["PATH"]
        options.append("-DGGML_VULKAN=ON")
    else:
        options.append("-DGGML_VULKAN=OFF")

    say("== build")
    build = WORK / "build"
    shutil.rmtree(build, ignore_errors=True)
    configure = ["cmake", "-S", source, "-B", build] + options
    if WINDOWS:
        configure += ["-A", "x64"]
    run(configure, env=env)
    run(["cmake", "--build", build, "--config", "Release", "--parallel", str(os.cpu_count() or 2)], env=env)
    bin_dir = build / "bin" / "Release" if WINDOWS else build / "bin"

    say("== package")
    stage = WORK / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    report = gather_windows(bin_dir, stage, variant) if WINDOWS else gather_unix(bin_dir, stage)
    write_licenses(source, stage / "licenses", report["runtime"])

    say("== the built server, asked")
    # A real model, so the answers have words and times: never packaged.
    model = obtain(pin["testModel"], cache)
    problems = smoke(stage, model, source / "samples" / "jfk.wav",
                     source / "models" / "for-tests-silero-v6.2.0-ggml.bin")
    if problems:
        fail(f"{len(problems)} check(s) of the built server failed")

    compiler = ""
    cache_file = build / "CMakeCache.txt"
    if cache_file.exists():
        found = re.search(r"^CMAKE_CXX_COMPILER:\w+=(.+)$", cache_file.read_text(encoding="utf-8", errors="replace"), re.M)
        compiler = found.group(1) if found else ""
    files = {p.relative_to(stage).as_posix(): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    info = {
        "tag": tag,
        "variant": variant,
        "platform": "win-x64" if WINDOWS else "linux-x64",
        "whisper": {k: pin["whisper"][k] for k in ("version", "tag", "url", "sha256")},
        "patch": {"file": "patches/whisper-server.patch", "sha256": sha256(PATCH)},
        "vulkanSdk": ({k: pin["vulkanSdk"][k] for k in ("version", "url", "sha256")} if variant == "vulkan" else None),
        "testModel": {k: pin["testModel"][k] for k in ("url", "sha256")},
        "cmakeOptions": options,
        "compiler": compiler,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "depends": report,
        "files": files,
    }
    (stage / "BUILD.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")

    name = f"whisper-runtime-{tag}-{variant}-x64.zip" if WINDOWS else f"whisper-runtime-{tag}-{variant}-linux-x64.zip"
    dest = DIST / name
    write_zip(stage, dest)
    with zipfile.ZipFile(dest) as check:
        names = check.namelist()
        if EXE not in names or check.testzip() is not None:
            fail(f"{name} is not a good archive")
    say(f"PASS  {name}: {dest.stat().st_size} bytes, {len(names)} files, SHA-256 {sha256(dest)}")
    say(f"BUILD: PASS ({int(time.time() - started)} s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("pin")
    pin.add_argument("--whisper", required=True, help="a tag of ggml-org/whisper.cpp, e.g. v1.9.2")
    pin.add_argument("--vulkan-sdk", help="a version of LunarG's Vulkan SDK for Windows, or `latest`")
    pin.add_argument("--cache", default=str(WORK / "sources"))
    pin.set_defaults(run=cmd_pin)
    build = commands.add_parser("build")
    build.add_argument("--variant", choices=("vulkan", "cpu"), default="vulkan" if WINDOWS else "cpu")
    build.add_argument("--tag")
    build.add_argument("--cache")
    build.set_defaults(run=cmd_build)
    check = commands.add_parser("smoke")
    check.add_argument("--dir", required=True)
    check.add_argument("--model", required=True, help="a real ggml model, e.g. ggml-tiny.bin")
    check.add_argument("--vad-model", required=True, help="the voice detection's ggml model")
    check.add_argument("--wav", required=True)
    check.set_defaults(run=cmd_smoke)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
