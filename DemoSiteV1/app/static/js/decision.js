/**
 * Decision Point Client – Phase 1 Stub
 * Finds all .widget-slot elements, calls /api/decision, and renders the response.
 * In Phase 1 the backend returns no-op so slots stay empty.
 */
(function () {
    const ENDPOINT = '/api/decision';

    document.addEventListener('DOMContentLoaded', () => {
        const slots = document.querySelectorAll('.widget-slot');
        slots.forEach(async (slot) => {
            const decisionPoint = slot.dataset.decisionPoint;
            let context = {};
            try {
                context = JSON.parse(slot.dataset.context || '{}');
            } catch (_) { }

            try {
                const res = await fetch(ENDPOINT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: window.SESSION_ID,
                        decision_point: decisionPoint,
                        context: context,
                    }),
                });
                const data = await res.json();

                if (data.action === 'no-op') {
                    // Phase 1: no widget rendered
                    return;
                }

                // Future: render widget based on data.action
                slot.innerHTML = renderWidget(data.action, context);
                slot.classList.add('widget-active');

                // Track widget impression
                trackWidgetEvent('widget_impression', decisionPoint, data.action);

                // Listen for interactions
                slot.addEventListener('click', (e) => {
                    if (e.target.closest('.widget-close')) {
                        slot.innerHTML = '';
                        slot.classList.remove('widget-active');
                        trackWidgetEvent('widget_dismiss', decisionPoint, data.action);
                    } else {
                        trackWidgetEvent('widget_click', decisionPoint, data.action);
                    }
                });
            } catch (_) { }
        });
    });

    function renderWidget(action, context) {
        const widgets = {
            trending_carousel: `
                <div class="alert alert-info d-flex justify-content-between align-items-center">
                    <span><i class="bi bi-fire"></i> <strong>Trending Now</strong> – Check out what others are buying!</span>
                    <button class="btn-close widget-close" aria-label="Close"></button>
                </div>`,
            discount_banner: `
                <div class="alert alert-warning d-flex justify-content-between align-items-center">
                    <span><i class="bi bi-tag"></i> <strong>Special Offer!</strong> Use code SAVE10 for 10% off.</span>
                    <button class="btn-close widget-close" aria-label="Close"></button>
                </div>`,
            frequently_bought_together: `
                <div class="alert alert-success d-flex justify-content-between align-items-center">
                    <span><i class="bi bi-people"></i> <strong>Frequently Bought Together</strong> – Customers also purchased these items.</span>
                    <button class="btn-close widget-close" aria-label="Close"></button>
                </div>`,
        };
        return widgets[action] || '';
    }

    function trackWidgetEvent(eventType, decisionPoint, action) {
        fetch('/api/events', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: window.SESSION_ID,
                event_type: eventType,
                page: window.location.pathname,
                element: 'widget-slot',
                metadata: { decision_point: decisionPoint, action: action },
            }),
            keepalive: true,
        }).catch(() => { });
    }
})();
