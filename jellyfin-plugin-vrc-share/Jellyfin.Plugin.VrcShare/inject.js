(function () {
    'use strict';

    // Adds a "VR Share Link" button to the item detail page's button row
    // (admins only). Clicking it asks the plugin's own backend to mint a
    // time-limited jellyfin-vrc-stream share link and copies it to the
    // clipboard. Uses `.mainDetailButtons` (the row holding Play/More/etc.,
    // see src/apps/legacy/controllers/itemDetails/index.html in jellyfin-web)
    // rather than the "..." overflow menu, since that container class has
    // been stable across jellyfin-web releases for a long time - the
    // overflow menu's internal item list is more likely to change shape.

    function getItemIdFromHash() {
        var match = window.location.hash.match(/[?&]id=([a-f0-9-]+)/i);
        return match ? match[1] : null;
    }

    function isDetailsPage() {
        return window.location.hash.indexOf('#/details') === 0;
    }

    // Material Symbols Outlined "head_mounted_device" (FILL 0, wght 400, GRAD
    // 0, opsz 24), inlined as SVG rather than relying on jellyfin-web's
    // bundled icon font: that font is the older, frozen "Material Icons" set
    // (material-design-icons-iconfont) and doesn't include this glyph.
    var HEAD_MOUNTED_DEVICE_SVG =
        '<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24" fill="currentColor" aria-hidden="true">' +
        '<path d="M300-240q-66 0-113-47t-47-113v-163q0-51 32-89.5t82-47.5q57-11 113-15.5t113-4.5q57 0 113.5 4.5T706-700q50 10 82 48t32 89v163q0 66-47 113t-113 47h-40q-13 0-26-1.5t-25-6.5l-64-22q-12-5-25-5t-25 5l-64 22q-12 5-25 6.5t-26 1.5h-40Zm0-80h40q7 0 13.5-1t12.5-3q29-9 56.5-19t57.5-10q30 0 58 9.5t56 19.5q6 2 12.5 3t13.5 1h40q33 0 56.5-23.5T740-400v-163q0-22-14-38t-35-21q-52-11-104.5-14.5T480-640q-54 0-106 4t-105 14q-21 4-35 20.5T220-563v163q0 33 23.5 56.5T300-320ZM40-400v-160h60v160H40Zm820 0v-160h60v160h-60Zm-380-80Z"/>' +
        '</svg>';

    function buildButton(itemId) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'button-flat btnVrcShare detailButton emby-button';
        btn.title = 'Create a time-limited VRChat share link';
        btn.innerHTML =
            '<div class="detailButton-content">' + HEAD_MOUNTED_DEVICE_SVG + '</div>';
        btn.addEventListener('click', function () {
            createShareLink(itemId, btn);
        });
        return btn;
    }

    function createShareLink(itemId, btn) {
        btn.disabled = true;
        btn.title = 'Creating share link…';

        var url = window.ApiClient.getUrl('VrcShare/CreateLink', { itemId: itemId });

        window.ApiClient.ajax({
            type: 'POST',
            url: url,
            dataType: 'json'
        }).then(function (result) {
            var minutes = Math.round((result.expires_at - Date.now() / 1000) / 60);
            return copyToClipboard(result.url).then(function () {
                notify('Share link copied! Valid for ~' + minutes + ' minutes.');
            });
        }).catch(function (err) {
            var message = (err && err.message) || 'Failed to create share link';
            notify(message, true);
        }).then(function () {
            btn.disabled = false;
            btn.title = 'Create a time-limited VRChat share link';
        });
    }

    function copyToClipboard(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(text);
        }
        // Fallback for contexts without the async clipboard API (e.g. non-HTTPS).
        var textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        try {
            document.execCommand('copy');
        } finally {
            document.body.removeChild(textarea);
        }
        return Promise.resolve();
    }

    function notify(message, isError) {
        if (window.Dashboard && typeof window.Dashboard.alert === 'function') {
            window.Dashboard.alert(message);
        } else {
            // eslint-disable-next-line no-alert
            window.alert(message);
        }
        if (isError) {
            console.error('[VrcShare]', message);
        }
    }

    function addButtonIfNeeded() {
        if (!isDetailsPage()) {
            return;
        }

        var itemId = getItemIdFromHash();
        if (!itemId) {
            return;
        }

        var container = document.querySelector('.mainDetailButtons');
        if (!container || container.querySelector('.btnVrcShare')) {
            return;
        }

        if (!window.ApiClient || typeof window.ApiClient.getCurrentUser !== 'function') {
            return;
        }

        window.ApiClient.getCurrentUser().then(function (user) {
            if (!user || !user.Policy || !user.Policy.IsAdministrator) {
                return;
            }
            // Re-check in case of a race with another invocation while the
            // user lookup was in flight.
            if (container.querySelector('.btnVrcShare')) {
                return;
            }
            var button = buildButton(itemId);
            var moreCommandsBtn = container.querySelector('.btnMoreCommands');
            if (moreCommandsBtn) {
                container.insertBefore(button, moreCommandsBtn);
            } else {
                container.appendChild(button);
            }
        }).catch(function () {
            // Not logged in yet, or request failed - just don't show the button.
        });
    }

    // jellyfin-web fires 'viewshow' on navigation between SPA views. Also
    // poll briefly on hash changes as a fallback, since detail pages can
    // finish rendering their button row slightly after the view event.
    document.addEventListener('viewshow', addButtonIfNeeded);
    window.addEventListener('hashchange', function () {
        setTimeout(addButtonIfNeeded, 300);
        setTimeout(addButtonIfNeeded, 1000);
    });
})();
