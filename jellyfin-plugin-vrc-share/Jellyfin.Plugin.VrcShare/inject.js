(function () {
    'use strict';

    // Adds a "VR Share Link" button to the item detail page's button row
    // (admins only). Clicking it asks the plugin's own backend to mint a
    // time-limited jellyfin-vrc-stream share link and copies it to the
    // clipboard. Uses `.detailButtons` (the row holding Play/More/etc.)
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

    function buildButton(itemId) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'button-flat btnVrcShare detailButton emby-button';
        btn.title = 'Create a time-limited VRChat share link';
        btn.innerHTML =
            '<span class="material-icons link" aria-hidden="true"></span>' +
            '<span class="detailButton-content"><span>VR Share Link</span></span>';
        btn.addEventListener('click', function () {
            createShareLink(itemId, btn);
        });
        return btn;
    }

    function setButtonLabel(btn, text) {
        var span = btn.querySelector('.detailButton-content span');
        if (span) {
            span.textContent = text;
        }
    }

    function createShareLink(itemId, btn) {
        btn.disabled = true;
        setButtonLabel(btn, 'Creating…');

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
            setButtonLabel(btn, 'VR Share Link');
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

        var container = document.querySelector('.detailButtons');
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
            container.appendChild(buildButton(itemId));
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
