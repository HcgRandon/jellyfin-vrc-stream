using Jellyfin.Plugin.VrcShare.Services;
using MediaBrowser.Controller;
using MediaBrowser.Controller.Plugins;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;

namespace Jellyfin.Plugin.VrcShare;

/// <summary>
/// Registers this plugin's services with Jellyfin's dependency injection container.
/// </summary>
public class PluginServiceRegistrator : IPluginServiceRegistrator
{
    /// <inheritdoc />
    public void RegisterServices(IServiceCollection serviceCollection, IServerApplicationHost applicationHost)
    {
        // Used by VrcShareController to call the proxy's /share endpoint server-side.
        serviceCollection.AddHttpClient();

        // Injects <script src="/VrcShare/inject.js"> into web index.html at request
        // time - see ScriptInjectionStartupFilter for why this approach is used
        // instead of writing to jellyfin-web's files on disk.
        serviceCollection.AddSingleton<IStartupFilter, ScriptInjectionStartupFilter>();
    }
}
