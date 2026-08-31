using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace Jellyfin.Plugin.VrcShare.Services;

/// <summary>
/// Injects a &lt;script src="/VrcShare/inject.js"&gt; tag into jellyfin-web's
/// index.html at request time via ASP.NET middleware.
///
/// This never writes to the web folder on disk (avoiding permission issues on
/// Docker installs) and survives jellyfin-web updates, since it only rewrites
/// the response on the way out. It is deliberately defensive: on any error it
/// serves the original response unchanged rather than breaking the web client.
/// </summary>
public class ScriptInjectionStartupFilter : IStartupFilter
{
    private const string ScriptTag = "<script src=\"/VrcShare/inject.js\" defer></script>";

    private readonly ILogger<ScriptInjectionStartupFilter> _logger;
    private int _loggedOnce;

    /// <summary>
    /// Initializes a new instance of the <see cref="ScriptInjectionStartupFilter"/> class.
    /// </summary>
    /// <param name="logger">Logger instance.</param>
    public ScriptInjectionStartupFilter(ILogger<ScriptInjectionStartupFilter> logger)
    {
        _logger = logger;
    }

    /// <inheritdoc />
    public Action<IApplicationBuilder> Configure(Action<IApplicationBuilder> next)
    {
        return app =>
        {
            // Registered before the rest of the pipeline so this runs outermost.
            app.Use(InvokeAsync);
            next(app);
        };
    }

    private async Task InvokeAsync(HttpContext context, Func<Task> nextMw)
    {
        if (!IsIndexRequest(context.Request.Path.Value) || !HttpMethods.IsGet(context.Request.Method))
        {
            await nextMw().ConfigureAwait(false);
            return;
        }

        var config = Plugin.Instance?.Configuration;
        if (config == null || config.DisableScriptInjectionMiddleware)
        {
            await nextMw().ConfigureAwait(false);
            return;
        }

        // Normalize the request so the static handler returns a complete, plain-text
        // 200 we can rewrite: drop compression and range negotiation.
        context.Request.Headers.Remove("Accept-Encoding");
        context.Request.Headers.Remove("Range");
        context.Request.Headers.Remove("If-Range");

        var originalBody = context.Response.Body;
        using var buffer = new MemoryStream();
        context.Response.Body = buffer;
        try
        {
            await nextMw().ConfigureAwait(false);
        }
        catch
        {
            context.Response.Body = originalBody;
            throw;
        }

        context.Response.Body = originalBody;
        buffer.Seek(0, SeekOrigin.Begin);

        var isHtml = context.Response.StatusCode == 200
            && (context.Response.ContentType?.Contains("text/html", StringComparison.OrdinalIgnoreCase) ?? false);

        if (!isHtml)
        {
            await buffer.CopyToAsync(originalBody).ConfigureAwait(false);
            return;
        }

        string html;
        using (var reader = new StreamReader(buffer, Encoding.UTF8, true, 1024, leaveOpen: true))
        {
            html = await reader.ReadToEndAsync().ConfigureAwait(false);
        }

        try
        {
            var alreadyInjected = html.Contains(ScriptTag, StringComparison.OrdinalIgnoreCase);
            var bodyClose = html.LastIndexOf("</body>", StringComparison.OrdinalIgnoreCase);

            if (!alreadyInjected && bodyClose >= 0)
            {
                html = html.Substring(0, bodyClose) + ScriptTag + "\n" + html.Substring(bodyClose);

                if (Interlocked.Exchange(ref _loggedOnce, 1) == 0)
                {
                    _logger.LogInformation("VRC Share: injected inject.js into web index.html via request-time middleware.");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "VRC Share script injection error (serving original HTML).");
        }

        var bytes = Encoding.UTF8.GetBytes(html);
        context.Response.ContentType = "text/html;charset=utf-8";
        context.Response.ContentLength = bytes.Length;
        // The body changed, so any validators set by the static-file handler are
        // no longer valid, and we don't support range requests on the rewritten doc.
        context.Response.Headers.Remove("ETag");
        context.Response.Headers.Remove("Last-Modified");
        context.Response.Headers.Remove("Accept-Ranges");
        await originalBody.WriteAsync(bytes, 0, bytes.Length).ConfigureAwait(false);
    }

    // Matches the web app shell however it is requested: bare "/web", "/web/"
    // (SPA serve), and explicit "/web/index.html". EndsWith keeps this correct
    // when Jellyfin is hosted under a base-url prefix (e.g. /jellyfin/web/).
    private static bool IsIndexRequest(string? path)
    {
        if (string.IsNullOrEmpty(path))
        {
            return false;
        }

        return path.EndsWith("/web/index.html", StringComparison.OrdinalIgnoreCase)
            || path.EndsWith("/web/", StringComparison.OrdinalIgnoreCase)
            || path.Equals("/web", StringComparison.OrdinalIgnoreCase);
    }
}
