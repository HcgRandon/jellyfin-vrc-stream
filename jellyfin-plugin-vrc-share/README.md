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

## Build

```bash
cd jellyfin-plugin-vrc-share
dotnet build -c Release
```

The compiled plugin is at
`Jellyfin.Plugin.VrcShare/bin/Release/net9.0/Jellyfin.Plugin.VrcShare.dll`.

## Install

1. Copy `Jellyfin.Plugin.VrcShare.dll` into a new folder under Jellyfin's
   plugin directory, e.g. `<jellyfin-config>/plugins/VRC Share_1.0.0.0/`.
2. Restart Jellyfin.
3. In the admin dashboard, go to **Plugins → VRC Share** and set:
   - **Proxy Base URL** - e.g. `https://stream.example.com`
   - **Proxy Admin API Key** - must match `ADMIN_API_KEY` on the proxy
   - **Default link lifetime** - defaults to 86400 seconds (24h)
4. Open any movie or episode as an administrator - a **VR Share Link** button
   appears next to Play. Click it, then paste the copied URL into your VRChat
   video player.

## How the button gets injected

Jellyfin's plugin SDK has no first-class "add a button to the item page"
extension point, so this plugin injects a small `<script>` tag into
`index.html` at request time via ASP.NET middleware (`IStartupFilter`) - the
same technique used by plugins like
[Jellyfin-JavaScript-Injector](https://github.com/n00bcodr/Jellyfin-JavaScript-Injector).
Nothing on disk is modified, so this survives jellyfin-web updates. The
injected script (`inject.js`) only adds the button into `.detailButtons` (the
row holding Play/More) rather than hooking Jellyfin's internal overflow-menu
item list, since `.detailButtons` has been a stable class across jellyfin-web
releases - the tradeoff is a dedicated button instead of a "⋮" menu entry.

If this ever conflicts with another script-injecting plugin, tick **Disable
the "VR Share Link" button injection** on the config page; the API endpoints
keep working regardless.
