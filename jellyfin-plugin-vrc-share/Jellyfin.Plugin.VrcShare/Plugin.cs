using System;
using System.Collections.Generic;
using Jellyfin.Plugin.VrcShare.Configuration;
using MediaBrowser.Common.Configuration;
using MediaBrowser.Common.Plugins;
using MediaBrowser.Model.Plugins;
using MediaBrowser.Model.Serialization;

namespace Jellyfin.Plugin.VrcShare;

/// <summary>
/// Adds a "VR Share Link" button to item detail pages (admins only) that mints
/// a time-limited jellyfin-vrc-stream share link and copies it to the clipboard.
/// </summary>
public class Plugin : BasePlugin<PluginConfiguration>, IHasWebPages
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Plugin"/> class.
    /// </summary>
    /// <param name="applicationPaths">Instance of the <see cref="IApplicationPaths"/> interface.</param>
    /// <param name="xmlSerializer">Instance of the <see cref="IXmlSerializer"/> interface.</param>
    public Plugin(IApplicationPaths applicationPaths, IXmlSerializer xmlSerializer)
        : base(applicationPaths, xmlSerializer)
    {
        Instance = this;
    }

    /// <inheritdoc />
    public override string Name => "VRC Share";

    /// <inheritdoc />
    public override Guid Id => Guid.Parse("a3f1e6d2-8b4c-4e9a-9c3f-2d7a5b6e1234");

    /// <summary>
    /// Gets the current plugin instance.
    /// </summary>
    public static Plugin? Instance { get; private set; }

    /// <inheritdoc />
    public IEnumerable<PluginPageInfo> GetPages()
    {
        yield return new PluginPageInfo
        {
            Name = "VrcShare",
            EmbeddedResourcePath = string.Format("{0}.Configuration.configPage.html", GetType().Namespace)
        };
    }
}
