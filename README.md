# whisper-build

Builds [whisper.cpp](https://github.com/ggml-org/whisper.cpp)'s
`whisper-server` for Windows (x64) and macOS (Apple Silicon) from sources
pinned by SHA-256, with one patch, as a package another program can start
and talk to on the same computer.

- `sources.pin.json` names the exact sources: whisper.cpp's source archive,
  the Vulkan SDK the Windows shaders are compiled with, and the small model
  the built server is asked with (it is never packaged), each with its
  SHA-256.
- `patches/whisper-server.patch` is the one change to the source (below).
- `build.py build` checks every source against its hash, applies the patch
  (it must apply exactly), and builds the server with ggml's backends as
  loadable libraries. It gathers the program and the libraries it needs,
  accounts for every library they link, writes the archive, unpacks it into
  a fresh folder, starts the server from there and checks the patch from
  outside.
- The workflow runs that on GitHub's `windows-2022` and `macos-26` runners.
  A pushed tag publishes one release with both packages, `SHA256SUMS` over
  both and `sources.pin.json`: a tag is both platforms, from one commit.

## The packages

**Windows x64** (`whisper-runtime-<tag>-vulkan-x64.zip`) holds, at its top,
`whisper-server.exe`, `whisper.dll`, the `ggml` libraries (`ggml-cpu-*.dll`,
one per CPU generation, and `ggml-vulkan.dll`), the Visual C++ runtime
libraries they need (copied from Visual Studio's redistributable folder, so
the package runs on a Windows that has none installed), `licenses/` and
`BUILD.json` (versions, options, hashes, what each file depends on). Only
`ggml-vulkan.dll` needs the Vulkan loader, which a GPU driver installs: on a
computer without one the server runs on the CPU.

**macOS arm64** (`whisper-runtime-<tag>-metal-macos-arm64.zip`) holds, at
its top, `whisper-server` (mode 0755), `libwhisper.1.dylib`,
`libggml.0.dylib`, `libggml-base.0.dylib`, the loadable backends
(`libggml-cpu-apple_m1.so`, `libggml-cpu-apple_m2_m3.so`,
`libggml-cpu-apple_m4.so`, `libggml-blas.so` for Apple's Accelerate, and
`libggml-metal.so`), `licenses/` and `BUILD.json`. Built with
`GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON` (the shaders are carried as
source and compiled by the Mac's own Metal framework at the server's first
start: nothing of Xcode is needed to run it), `GGML_BLAS=ON` with Apple's
Accelerate, `GGML_OPENMP=OFF` (Apple's compiler has no OpenMP), Core ML off,
arm64 only, macOS 13.3 at the least, and `@loader_path` as the only run
path, so the package runs from any folder with no environment variable. The
build checks every file: arm64 only, its lowest macOS, its run paths, every
library it links (a file of the package or a part of macOS, nothing else),
and the linker's own signature (`codesign --verify --strict`). Only
`libggml-metal.so` links Metal: on a Mac whose Metal fails the server runs
on the CPU.

**The macOS package is not notarized.** Each file carries the linker's
ad-hoc signature, which Apple Silicon requires and accepts. macOS checks a
download only when it carries the quarantine mark, which a web browser sets
and `curl` does not. Downloaded by browser, clear the mark on the unpacked
folder before the first start:

```
xattr -dr com.apple.quarantine <folder>
```

## The patch

Upstream's server is made to be reached by anything on the computer: it has
no sign-in, answers every web page (`Access-Control-Allow-Origin: *`), and
its `/load` route exits the whole server when it is sent a file that is not
a model. The patch makes it answer one program only, the one that started
it:

- **A token.** The server reads `PBH_WHISPER_TOKEN` from its environment and
  refuses to start without one of at least 32 characters (exit code 4).
  Every request must carry `Authorization: Bearer <token>`; anything else is
  answered `401` before its body is read.
- **No web pages.** A request with an `Origin` header is answered `403`,
  token or not, and no answer carries a cross-origin header.
- **No `/load`.** The model is the one the server was started with.
- **A line on standard error** once the port is held:
  `[pbh] listening on <host>:<port>`, so the program that started it can
  tell its own server from anything else on that port.
- **Each word's times on the recording's clock.** Upstream writes a word's
  start and end as the decoder saw them, on the timeline voice detection
  has cut the silence out of, while a segment's are mapped back to the
  recording: after removed silence every word came early by the silence
  removed, outside its own segment. The patch writes them through the
  library's own mapping (`whisper_full_get_token_t0` and `_t1`).

**A request's times are its own.** Before whisper.cpp 1.9.6 the library
kept the time table of the last request that had voice detection on, so a
request without it that followed one with it had its segments' times mapped
through the earlier recording's table (late by the silence that recording
had lost). This patch mended it in the library until 1.9.6; since 1.9.6
(whisper.cpp's PR #4122) upstream clears that state at the start of every
request without voice detection, in both calls, and the patch changes the
server only. `build.py` refuses a library that holds a `[pbh]` place or
does not clear it in both calls.

Every changed place is marked `[pbh]` in the source. `build.py` refuses a
patch that does not apply exactly, and checks each of the points above
against the built server before it lets an archive stand: for the last one
it asks the server for a recording with 20 seconds of silence before it and
voice detection on, then for the recording itself with voice detection off,
and wants every segment of the second answer inside its own recording.

**Each word's times** are checked too, on both platforms: the form a
program sends for one part (`verbose_json`, no language probabilities,
language `auto`, no translation, voice detection on) is sent for the
recording and for the recording with the silence before it, and every
segment of each answer must carry `words`, each with a start no later than
its end, inside its segment.

**The check build** of the releases before 1.9.6 (`--unfixed-library`, the
library without the old fix) is gone with that fix: its checks passed only
when the check of the times failed, and on 1.9.6 they would not. The check
of the times stays in every build. `build.py smoke --expect-times-fault`
still asks an unpacked build of an older whisper.cpp that way (exit code 3:
the times were right after all).

## Moving a pin

```
python build.py pin --whisper <tag> --vulkan-sdk <version>
```

writes `sources.pin.json` from fresh downloads (`--vulkan-sdk latest` takes
the newest). If the patch no longer applies to the new source, it is made
again against it: `build.py build` says so before anything is built. Commit
and push a tag named `<whisper.cpp version>-<build number>` (`1.9.2-1`). A
build that fails leaves its tag used; the next try counts up.

`python build.py build --variant cpu` builds without a GPU and runs on
Linux too; nothing is published from it. It is how the script and the patch
are tried where there is no Windows or Mac.

`python build.py smoke --dir <unpacked package> --model <ggml model>
--vad-model <ggml model> --wav <file> [--want-gpu metal|vulkan]` asks a
package that is already built (Python 3.9 or newer, standard library only).
`python build.py bench ... [--no-gpu]` times one request of a program's own
form: a report.

## Licence

The build scripts in this repository are under the Apache License 2.0
(`LICENSE`). The patch is offered under the MIT licence of the file it
changes. The sources keep their own licences; each archive carries them in
`licenses/`.
