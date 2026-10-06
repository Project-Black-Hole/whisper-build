# whisper-build

Builds [whisper.cpp](https://github.com/ggml-org/whisper.cpp)'s
`whisper-server` for Windows (x64) from sources pinned by SHA-256, with one
patch, as a package another program can start and talk to on this computer.

- `sources.pin.json` names the exact sources: whisper.cpp's source archive,
  the Vulkan SDK the shaders are compiled with, and the small model the
  built server is asked with (it is never packaged), each with its SHA-256.
- `patches/whisper-server.patch` is the one change to the source (below).
- `build.py build` checks every source against its hash, applies the patch
  (it must apply exactly), and builds the server with ggml's backends as
  loadable libraries: every CPU variant, and Vulkan. It gathers the program
  and the libraries it needs, accounts for every library they import, starts
  the built server and checks the patch from outside, and writes
  `dist/whisper-runtime-<tag>-vulkan-x64.zip`.
- The workflow runs that on GitHub's `windows-2022` runner. A pushed tag
  publishes a release: the archive, `SHA256SUMS` and `sources.pin.json`.

An archive holds, at its top, `whisper-server.exe`, `whisper.dll`, the `ggml`
libraries (`ggml-cpu-*.dll`, one per CPU generation, and `ggml-vulkan.dll`),
the Visual C++ runtime libraries they need (copied from Visual Studio's
redistributable folder, so the package runs on a Windows that has none
installed), `licenses/` and `BUILD.json` (versions, options, hashes, what
each file depends on). Only `ggml-vulkan.dll` needs the Vulkan loader, which
a GPU driver installs: on a computer without one the server runs on the CPU.

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

It also mends one fault of the library itself (`src/whisper.cpp`):

- **A request's times are its own.** Upstream keeps the time table of the
  last request that had voice detection on, and clears it only when the
  next request has voice detection on too. A request without voice
  detection that followed one with it had its segments' times mapped
  through the earlier recording's table: late by however much silence that
  recording had lost, and past the end of its own sound. The patch clears
  the table, the detected stretches and their flag at the start of every
  request without voice detection.

Every changed place is marked `[pbh]` in the source. `build.py` refuses a
patch that does not apply exactly, and checks each of the points above
against the built server before it writes an archive: for the last one it
asks the server for a recording with 20 seconds of silence before it and
voice detection on, then for the recording itself with voice detection off,
and wants every segment of the second answer inside its own recording.

## Moving a pin

```
python build.py pin --whisper <tag> --vulkan-sdk <version>
```

writes `sources.pin.json` from fresh downloads (`--vulkan-sdk latest` takes
the newest). If the patch no longer applies to the new source, it is made
again against it: `build.py build` says so before anything is built. Commit
and push a tag named `<whisper.cpp version>-<build number>` (`1.9.2-1`).

`python build.py build --variant cpu` builds without Vulkan and runs on
Linux too; nothing is published from it. It is how the script and the patch
are tried where there is no Windows.

## Licence

The build scripts in this repository are under the Apache License 2.0
(`LICENSE`). The patch is offered under the MIT licence of the file it
changes. The sources keep their own licences; each archive carries them in
`licenses/`.
