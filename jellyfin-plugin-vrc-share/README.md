# VRC Share (Jellyfin plugin)

Adds a **VR Share Link** button to movie/episode detail pages in Jellyfin's web
UI. Click it as an administrator and it:

1. Calls this plugin's own backend (`POST /VrcShare/CreateLink`), authenticated
   by your existing Jellyfin admin session - no extra login.
2. The backend calls the [jellyfin-vrc-stream](../) proxy's `POST /share`
   endpoint, using the admin key from this plugin's configuration (never sent
   to the browser).
3. The resulting time-limited playback URL is copied to your clipboard, ready
   to paste into a VRChat video player.

## Requirements

- Jellyfin 10.11.x
- A jellyfin-vrc-stream proxy (this repo) with `ADMIN_API_KEY` set

## Install via plugin repository (recommended)

This plugin is distributed as a Jellyfin **plugin repository** - a
`manifest.json` that Jellyfin's own Plugins UI can read, so you get install
and update prompts like any catalog plugin, without touching files by hand.

1. In Jellyfin, go to **Dashboard → Plugins → Repositories → Add Repository**.
2. Add:
   - **Repository Name:** `VRC Share`
   - **Repository URL:** `https://raw.githubusercontent.com/C9Glax/jellyfin-vrc-stream/main/jellyfin-plugin-vrc-share/manifest.json`
3. Go to **Catalog**, find **VRC Share** under General, and install it.
4. Restart Jellyfin.
5. Continue at [Configure](#configure) below.

New versions are published as GitHub Releases (tag `plugin-vX.Y.Z.W`) and
appear in the manifest automatically - see
[`.github/workflows/plugin-vrc-share-release.yaml`](../.github/workflows/plugin-vrc-share-release.yaml).

## Install manually (development / no repository)

```bash
cd jellyfin-plugin-vrc-share
dotnet build -c Release
```

The compiled plugin is at
`Jellyfin.Plugin.VrcShare/bin/Release/net9.0/Jellyfin.Plugin.VrcShare.dll`.

1. Copy `Jellyfin.Plugin.VrcShare.dll` into a new folder under Jellyfin's
   plugin directory, e.g. `<jellyfin-config>/plugins/VRC Share_1.0.0.0/`.
2. Restart Jellyfin.
3. Continue at [Configure](#configure) below.

## Configure

1. On the proxy itself, make sure `ADMIN_API_KEY` is actually set (and the
   proxy restarted) - if it isn't, every request to `/share` fails with
   `401: Admin API key is not configured on the server`, regardless of what
   the plugin sends.
2. In the admin dashboard, go to **Plugins → VRC Share** and set:
   - **Proxy Base URL** - e.g. `https://stream.example.com`
   - **Proxy Admin API Key** - must match `ADMIN_API_KEY` on the proxy exactly
   - **Default link lifetime** - defaults to 86400 seconds (24h)
3. Open any movie or episode as an administrator - a headset icon button
   appears next to Play. Click it, then paste the copied URL into your VRChat
   video player.

## How the button gets injected

Jellyfin's plugin SDK has no first-class "add a button to the item page"
extension point, so this plugin injects a small `<script>` tag into
`index.html` at request time via ASP.NET middleware (`IStartupFilter`) - the
same technique used by plugins like
[Jellyfin-JavaScript-Injector](https://github.com/n00bcodr/Jellyfin-JavaScript-Injector).
Nothing on disk is modified, so this survives jellyfin-web updates. The
injected script (`inject.js`) only adds the button into `.mainDetailButtons` (the
row holding Play/More) rather than hooking Jellyfin's internal overflow-menu
item list, since `.mainDetailButtons` has been a stable class across jellyfin-web
releases - the tradeoff is a dedicated button instead of a "⋮" menu entry.

If this ever conflicts with another script-injecting plugin, tick **Disable
the "VR Share Link" button injection** on the config page; the API endpoints
keep working regardless.

## Cutting a release (maintainers)

1. Bump the version however you like (the release workflow overwrites
   `AssemblyVersion`/`FileVersion` in the `.csproj` from the tag anyway).
2. Create and push a tag named `plugin-vX.Y.Z.W`, e.g.:
   ```bash
   git tag plugin-v1.1.0.0
   git push origin plugin-v1.1.0.0
   ```
   That's it - no manual "Publish release" step needed.
3. [`plugin-vrc-share-release.yaml`](../.github/workflows/plugin-vrc-share-release.yaml)
   builds the plugin, creates the GitHub Release for that tag (with
   auto-generated notes) with `Jellyfin.Plugin.VrcShare_X.Y.Z.W.zip`
   attached, and commits a new entry to `manifest.json` - anyone with this
   repository added in Jellyfin sees the update automatically.
