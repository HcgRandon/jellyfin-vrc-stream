using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Reflection;
using System.Threading.Tasks;
using MediaBrowser.Common.Api;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace Jellyfin.Plugin.VrcShare.Api;

/// <summary>
/// Server-side endpoints backing the "VR Share Link" button injected into item
/// detail pages, plus the injector script itself.
/// </summary>
[ApiController]
[Route("VrcShare")]
public class VrcShareController : ControllerBase
{
    private readonly IHttpClientFactory _httpClientFactory;

    /// <summary>
    /// Initializes a new instance of the <see cref="VrcShareController"/> class.
    /// </summary>
    /// <param name="httpClientFactory">Factory used to call the jellyfin-vrc-stream proxy.</param>
    public VrcShareController(IHttpClientFactory httpClientFactory)
    {
        _httpClientFactory = httpClientFactory;
    }

    /// <summary>
    /// Serves the client-side injector script embedded in this assembly. Must
    /// stay unauthenticated - it needs to load on every page, including the
    /// login page - the button it adds only appears for administrators.
    /// </summary>
    /// <returns>The injector script.</returns>
    [HttpGet("inject.js")]
    [AllowAnonymous]
    [Produces("application/javascript")]
    public ActionResult GetInjectScript()
    {
        var assembly = Assembly.GetExecutingAssembly();
        // Embedded resource logical names are rooted at the assembly's RootNamespace
        // ("Jellyfin.Plugin.VrcShare"), not this controller's own namespace.
        var resourceName = $"{typeof(Plugin).Namespace}.inject.js";
        var stream = assembly.GetManifestResourceStream(resourceName);
        if (stream == null)
        {
            return NotFound();
        }

        Response.Headers["Cache-Control"] = "no-cache";
        return new FileStreamResult(stream, "application/javascript");
    }

    /// <summary>
    /// Mints a time-limited share link for a single media item by calling the
    /// jellyfin-vrc-stream proxy's POST /share endpoint server-side, so the
    /// proxy's admin key never reaches the browser. Requires an elevated
    /// (administrator) Jellyfin session - the same one already authenticating
    /// this request, no extra login needed.
    /// </summary>
    /// <param name="itemId">Jellyfin media item ID to share.</param>
    /// <param name="mode">"vod" or "live" (defaults to "vod").</param>
    /// <param name="ttlSeconds">Optional link lifetime override in seconds.</param>
    /// <returns>The proxy's JSON response, containing the share URL and expiry.</returns>
    [HttpPost("CreateLink")]
    [Authorize(Policy = Policies.RequiresElevation)]
    public async Task<ActionResult> CreateLink(
        [FromQuery] string itemId,
        [FromQuery] string mode = "vod",
        [FromQuery] int? ttlSeconds = null)
    {
        if (string.IsNullOrWhiteSpace(itemId))
        {
            return BadRequest("itemId is required");
        }

        var config = Plugin.Instance?.Configuration;
        if (config == null || string.IsNullOrWhiteSpace(config.ProxyBaseUrl) || string.IsNullOrWhiteSpace(config.AdminApiKey))
        {
            return Problem(
                "VRC Share plugin is not configured. Set Proxy Base URL and Admin API Key on the plugin's settings page.",
                statusCode: 500);
        }

        var client = _httpClientFactory.CreateClient();
        var payload = new
        {
            media_id = itemId,
            mode,
            ttl_seconds = ttlSeconds ?? config.DefaultTtlSeconds
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, $"{config.ProxyBaseUrl.TrimEnd('/')}/share")
        {
            Content = JsonContent.Create(payload)
        };
        request.Headers.Add("X-Admin-Key", config.AdminApiKey);

        HttpResponseMessage response;
        try
        {
            response = await client.SendAsync(request).ConfigureAwait(false);
        }
        catch (HttpRequestException ex)
        {
            return Problem($"Failed to reach the proxy at {config.ProxyBaseUrl}: {ex.Message}", statusCode: 502);
        }

        var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            return Problem($"Proxy returned {(int)response.StatusCode}: {body}", statusCode: 502);
        }

        // Pass the proxy's JSON straight through - no need to re-model it here,
        // and this keeps the two sides from drifting out of sync on field names.
        return Content(body, "application/json");
    }
}
