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
                page_depth: parseInt(
                    sessionStorage.getItem(`page_depth:${window.SESSION_ID || 'anonymous'}`) || '1',
                    10,
                ),
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
    // Only 25 (analytics-only, reward 0) and 50 (the single rewarded
    // engagement event) are emitted.  The simulator generates at most one
    // scroll_depth event per step, so emitting 75/100 as further scroll_depth
    // events would award the +0.2 engagement bonus repeatedly per page.
    const scrollThresholds = [25, 50];
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
    // Emitted once per page view at the 30 s threshold (matching the
    // simulator's at-most-one dwell event per step); the timer stops after
    // the first emission so long stays cannot accumulate repeated bonuses.
    const pageStart = Date.now();
    let dwellFired = false;
    const dwellTimer = setInterval(() => {
        dwellFired = true;
        sendEvent('dwell_time', { seconds: 30 });
        clearInterval(dwellTimer);
    }, 30000);

    // On page leave, record the actual dwell for analytics only when the
    // rewarded threshold event never fired (seconds < 30 → reward 0).
    window.addEventListener('beforeunload', () => {
        clearInterval(dwellTimer);
        if (!dwellFired) {
            const totalSeconds = Math.round((Date.now() - pageStart) / 1000);
            sendEvent('dwell_time', { seconds: Math.min(totalSeconds, 29), final: true });
        }
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

    // --- Checkout Submission ---
    const checkoutForm = document.getElementById('checkout-form');
    if (checkoutForm) {
        checkoutForm.addEventListener('submit', () => {
            sendEvent('checkout_submit', { element: 'checkout-form' });
        });
    }
})();
