/**
 * Event Tracking – Phase 1 Heuristic Data Collection
 * Automatically tracks: page_view, scroll_depth, dwell_time, click, exit_intent
 * All events sent to POST /api/events
 */
(function () {
    const ENDPOINT = '/api/events';

    function sendEvent(eventType, extra = {}) {
        const payload = {
            session_id: window.SESSION_ID,
            event_type: eventType,
            page: window.location.pathname,
            element: extra.element || '',
            metadata: {
                referrer: document.referrer,
                user_agent: navigator.userAgent,
                screen_width: window.screen.width,
                screen_height: window.screen.height,
                timestamp_client: new Date().toISOString(),
                ...extra,
            },
        };
        fetch(ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            keepalive: true,
        }).catch(() => { }); // fire-and-forget
    }

    // --- Page View ---
    sendEvent('page_view');

    // --- Scroll Depth ---
    const scrollThresholds = [25, 50, 75, 100];
    const scrollFired = new Set();

    function checkScroll() {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        if (docHeight <= 0) return;
        const pct = Math.round((scrollTop / docHeight) * 100);
        for (const t of scrollThresholds) {
            if (pct >= t && !scrollFired.has(t)) {
                scrollFired.add(t);
                sendEvent('scroll_depth', { depth: t });
            }
        }
    }
    window.addEventListener('scroll', checkScroll, { passive: true });

    // --- Dwell Time ---
    const pageStart = Date.now();
    let dwellIntervals = 0;
    const dwellTimer = setInterval(() => {
        dwellIntervals++;
        sendEvent('dwell_time', { seconds: dwellIntervals * 30 });
    }, 30000);

    // Send final dwell time on page leave
    window.addEventListener('beforeunload', () => {
        const totalSeconds = Math.round((Date.now() - pageStart) / 1000);
        sendEvent('dwell_time', { seconds: totalSeconds, final: true });
        clearInterval(dwellTimer);
    });

    // --- Click Tracking (product cards, buttons, links) ---
    document.addEventListener('click', (e) => {
        const target = e.target.closest('a, button, [data-track]');
        if (!target) return;
        const label =
            target.dataset.track ||
            target.textContent.trim().substring(0, 80) ||
            target.tagName;
        sendEvent('click', {
            element: target.tagName.toLowerCase(),
            text: label,
            href: target.href || '',
        });
    });

    // --- Exit Intent (mouse leaves viewport at top) ---
    let exitFired = false;
    document.addEventListener('mouseout', (e) => {
        if (exitFired) return;
        if (e.clientY <= 0) {
            exitFired = true;
            sendEvent('exit_intent');
        }
    });
})();
