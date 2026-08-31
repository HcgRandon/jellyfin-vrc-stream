using MediaBrowser.Model.Plugins;

namespace Jellyfin.Plugin.VrcShare.Configuration;

/// <summary>
/// Configuration for the VRC Share plugin, mirroring the settings on the
/// jellyfin-vrc-stream proxy that this plugin talks to.
/// </summary>
public class PluginConfiguration : BasePluginConfiguration
{
    /// <summary>
    /// Gets or sets the base URL of the jellyfin-vrc-stream proxy, e.g.
    /// "https://stream.example.com". Used to call POST /share.
    /// </summary>
    public string ProxyBaseUrl { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the admin API key configured on the proxy (JELLYFIN_API_KEY).
    /// Sent as the X-Admin-Key header when creating share links. Never exposed
    /// to the browser - only used server-side by <see cref="Api.VrcShareController"/>.
    /// </summary>
    public string AdminApiKey { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the default share link lifetime in seconds, used when the
    /// caller doesn't request a specific TTL.
    /// </summary>
    public int DefaultTtlSeconds { get; set; } = 86400;

    /// <summary>
    /// Gets or sets a value indicating whether the request-time script
    /// injection middleware is disabled. Useful as a kill switch if it ever
    /// interferes with another plugin's own index.html injection.
    /// </summary>
    public bool DisableScriptInjectionMiddleware { get; set; } = false;
}
