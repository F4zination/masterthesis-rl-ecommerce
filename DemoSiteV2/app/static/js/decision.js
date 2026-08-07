/**
 * Decision Point Client – Phase 2 Contextual Bandit
 * Requests decisions with context features and tracks response-linked widget outcomes.
 *
 * Event semantics mirror the CustomerSimulation training environment: per
 * decision at most ONE widget interaction event is emitted (widget_click XOR
 * widget_dismiss, whichever happens first), matching the simulator's
 * mutually-exclusive one-shot click/dismiss sampling.
 */
(function () {
    const ENDPOINT = '/api/decision';

    // decision_id -> true once a widget_click/widget_dismiss was emitted.
    const interactionSent = {};

    function sessionStorageKey(name) {
        return `${name}:${window.SESSION_ID || 'anonymous'}`;
    }

    function getPageDepth() {
        const key = sessionStorageKey('page_depth');
        const current = parseInt(sessionStorage.getItem(key) || '0', 10) || 0;
        const next = current + 1;
        sessionStorage.setItem(key, String(next));
        return next;
    }

    function getInitialReferrer() {
        const key = sessionStorageKey('initial_referrer');
        const stored = sessionStorage.getItem(key);
        if (stored !== null) {
            return stored;
        }
        const initial = document.referrer || '';
        sessionStorage.setItem(key, initial);
        return initial;
    }

    document.addEventListener('DOMContentLoaded', () => {
        const pageDepth = getPageDepth();
        const initialReferrer = getInitialReferrer();

        // Scroll-depth 70% decision point — runs once per page load
        initScrollDecision(pageDepth);

        const slots = document.querySelectorAll('.widget-slot');
        slots.forEach(async (slot) => {
            const decisionPoint = slot.dataset.decisionPoint;
            let context = {};
            try {
                context = JSON.parse(slot.dataset.context || '{}');
            } catch (_) { }

            context = {
                ...context,
                page_depth: pageDepth,
                referrer: initialReferrer,
                screen_width: window.screen.width,
                screen_height: window.screen.height,
            };

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
                if (!res.ok) {
                    trackDecisionError(decisionPoint, 'http_error', res.status);
                    return;
                }
                const data = await res.json();

                if (data.action === 'no-op') {
                    return;
                }

                // renderWidget is async (trending_carousel fetches products)
                const html = await renderWidget(data.action, context);
                if (!html) {
                    // Nothing rendered — do NOT log an impression, otherwise the
                    // logged action would not match the intervention actually shown.
                    return;
                }
                slot.innerHTML = html;
                slot.classList.add('widget-active');

                // Track widget impression
                trackWidgetEvent('widget_impression', decisionPoint, data.action, data.decision_id, data.propensity);

                wireWidgetContainer(slot, decisionPoint, data.action, data.decision_id, data.propensity);
            } catch (_) {
                trackDecisionError(decisionPoint, 'network_or_parse_error', 0);
            }
        });
    });

    function trackDecisionError(decisionPoint, reasonCode, statusCode) {
        fetch('/api/events', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: window.SESSION_ID,
                event_type: 'opportunity_error',
                page: window.location.pathname,
                element: '',
                metadata: {
                    decision_point: decisionPoint,
                    reason_code: reasonCode,
                    http_status: statusCode,
                },
            }),
            keepalive: true,
        }).catch(() => { });
    }

    /**
     * Attach the shared interaction handlers to a rendered widget container:
     * action buttons, help options, close/dismiss, and generic clicks
     * (including recommendation links — anchors count as widget clicks).
     */
    function wireWidgetContainer(container, decisionPoint, action, decisionId, propensity) {
        wireWidgetActions(container, decisionPoint, action, decisionId, propensity);

        container.addEventListener('click', (e) => {
            if (e.target.closest('.widget-close')) {
                container.innerHTML = '';
                container.classList.remove('widget-active');
                trackWidgetInteraction('widget_dismiss', decisionPoint, action, decisionId, propensity);
            } else if (!e.target.closest('.widget-action-btn')) {
                // Anchors (recommendation links) deliberately included: a
                // product click on a carousel is the widget's success signal.
                trackWidgetInteraction('widget_click', decisionPoint, action, decisionId, propensity);
            }
        });
    }

    // ── Widget rendering ────────────────────────────────────────────────────

    async function renderWidget(action, context) {
        if (action === 'trending_carousel') {
            return await renderTrendingCarousel();
        }
        if (action === 'frequently_bought_together') {
            return await renderFrequentlyBoughtTogether(context);
        }
        const widgets = {
            discount_banner: `
                <div class="alert alert-warning mb-0" role="alert">
                    <div class="d-flex justify-content-between align-items-start gap-2">
                        <div>
                            <div class="fw-semibold mb-1"><i class="bi bi-tag-fill me-1"></i>Special Offer – 10% Off Your Order</div>
                            <div class="small text-muted mb-2">Code <strong>SAVE10</strong> — applied instantly to your cart.</div>
                            <button class="btn btn-sm btn-warning widget-apply-discount fw-semibold">
                                <i class="bi bi-check-lg me-1"></i>Apply Discount
                            </button>
                        </div>
                        <button class="btn-close widget-close flex-shrink-0" aria-label="Close"></button>
                    </div>
                </div>`,
            trust_badge: `
                <div class="alert alert-light border border-success mb-0 p-3" role="alert">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <div class="fw-semibold text-success">
                            <i class="bi bi-shield-check me-1"></i>Secure & Trusted Checkout
                        </div>
                        <button class="btn-close widget-close flex-shrink-0" aria-label="Close"></button>
                    </div>
                    <img
                        src="/static/images/trust_badge.png"
                        alt="DemoShop secure checkout: encrypted payments and 30-day returns"
                        class="img-fluid mb-2"
                        style="max-height: 180px; width: 100%; object-fit: contain; background: transparent;"
                        onerror="this.style.display='none'; this.nextElementSibling.classList.remove('d-none');"
                    >
                    <div class="small text-muted d-none mb-2">
                        Trust badge image unavailable.
                    </div>
                    <div class="small text-muted lh-sm">
                        <div><i class="bi bi-lock-fill me-1"></i>SSL encrypted payment processing</div>
                        <div><i class="bi bi-arrow-counterclockwise me-1"></i>30-day money-back guarantee</div>
                        <div><i class="bi bi-box2-heart me-1"></i>Free returns on all orders</div>
                    </div>
                </div>`,
            help_popup: `
                <div class="alert alert-light border mb-0 p-3" role="alert">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span class="fw-semibold small"><i class="bi bi-headset me-1"></i>Need Help?</span>
                        <button class="btn-close widget-close flex-shrink-0" aria-label="Close"></button>
                    </div>
                    <div class="list-group list-group-flush">
                        <button class="list-group-item list-group-item-action small help-opt widget-action-btn py-2">
                            <i class="bi bi-chat-dots me-2 text-primary"></i>Live Chat
                        </button>
                        <button class="list-group-item list-group-item-action small help-opt widget-action-btn py-2">
                            <i class="bi bi-book me-2 text-success"></i>Browse FAQ
                        </button>
                        <button class="list-group-item list-group-item-action small help-opt widget-action-btn py-2">
                            <i class="bi bi-envelope me-2 text-warning"></i>Contact Support
                        </button>
                    </div>
                </div>`,
        };
        return widgets[action] || '';
    }

    async function renderTrendingCarousel() {
        let products = [];
        try {
            const r = await fetch('/api/trending');
            products = await r.json();
        } catch (_) {
            return '';
        }
        const cards = products.map(p => `
            <a href="/product/${p.slug}" class="text-decoration-none text-dark flex-shrink-0"
               style="width:110px;">
                <div class="card h-100 border-0 shadow-sm">
                    <img src="${p.thumb_url || p.image_url}" alt="${escHtml(p.name)}" loading="lazy"
                         class="card-img-top rounded-top" style="height:80px;object-fit:cover;">
                    <div class="card-body p-1 text-center">
                        <div class="small fw-semibold lh-sm" style="font-size:.72rem;">${escHtml(p.name)}</div>
                        <div class="text-muted" style="font-size:.7rem;">&euro;${p.price.toFixed(2)}</div>
                    </div>
                </div>
            </a>`).join('');

        return `
            <div class="alert alert-info mb-0 p-2" role="alert">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="fw-semibold small"><i class="bi bi-fire me-1"></i>Trending Now</span>
                    <button class="btn-close btn-close-sm widget-close" aria-label="Close"></button>
                </div>
                <div class="d-flex gap-2 overflow-auto pb-1">
                    ${cards}
                </div>
            </div>`;
    }

    async function renderFrequentlyBoughtTogether(context) {
        let products = [];
        // Anchor the recommendation on the product being viewed so the widget shows
        // items that actually go with it; without an anchor the server falls back to
        // the session's cart contents.
        const anchorId = context && context.product_id;
        const query = anchorId ? `?product_id=${encodeURIComponent(anchorId)}` : '';
        try {
            const r = await fetch(`/api/frequently_bought_together${query}`);
            products = await r.json();
        } catch (_) {
            return '';
        }
        if (!Array.isArray(products) || products.length === 0) {
            return '';
        }
        const cards = products.map(p => `
            <a href="/product/${p.slug}" class="text-decoration-none text-dark flex-shrink-0"
               style="width:110px;">
                <div class="card h-100 border-0 shadow-sm">
                    <img src="${p.thumb_url || p.image_url}" alt="${escHtml(p.name)}" loading="lazy"
                         class="card-img-top rounded-top" style="height:80px;object-fit:cover;">
                    <div class="card-body p-1 text-center">
                        <div class="small fw-semibold lh-sm" style="font-size:.72rem;">${escHtml(p.name)}</div>
                        <div class="text-muted" style="font-size:.7rem;">&euro;${p.price.toFixed(2)}</div>
                    </div>
                </div>
            </a>`).join('');

        return `
            <div class="alert alert-success mb-0 p-2" role="alert">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="fw-semibold small"><i class="bi bi-people me-1"></i>Frequently Bought Together</span>
                    <button class="btn-close btn-close-sm widget-close" aria-label="Close"></button>
                </div>
                <div class="d-flex gap-2 overflow-auto pb-1">
                    ${cards}
                </div>
            </div>`;
    }

    function escHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // ── Widget action wiring ────────────────────────────────────────────────

    function wireWidgetActions(slot, decisionPoint, action, decisionId, propensity) {
        if (action === 'discount_banner') {
            const btn = slot.querySelector('.widget-apply-discount');
            if (!btn) return;
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                trackWidgetInteraction('widget_click', decisionPoint, action, decisionId, propensity);
                btn.disabled = true;
                btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Applying…';
                try {
                    const res = await fetch('/api/cart/discount', { method: 'POST' });
                    const data = await res.json();
                    if (data.status === 'already_applied') {
                        btn.innerHTML = '<i class="bi bi-check-circle me-1"></i>Already applied';
                    } else {
                        btn.innerHTML = '<i class="bi bi-check-circle-fill me-1"></i>Discount applied!';
                        btn.classList.replace('btn-warning', 'btn-success');
                        // Update visible totals on cart/checkout page without full reload
                        applyDiscountToPage(data);
                    }
                } catch (_) {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="bi bi-exclamation-triangle me-1"></i>Try again';
                }
            });
        }
        if (action === 'help_popup') {
            slot.querySelectorAll('.help-opt').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    trackWidgetInteraction('widget_click', decisionPoint, action, decisionId, propensity);
                    const icon = btn.querySelector('i').outerHTML;
                    btn.innerHTML = icon + ' ' + btn.textContent.trim() + ' <i class="bi bi-check-circle-fill text-success ms-1"></i>';
                    btn.disabled = true;
                });
            });
        }
    }

    /** Patch the cart/checkout total display in-place after discount is applied. */
    function applyDiscountToPage(data) {
        // cart page
        const subtotalEl = document.getElementById('cart-subtotal');
        const totalEl = document.getElementById('cart-total');
        const discountRow = document.getElementById('discount-row');
        const discountAmt = document.getElementById('discount-amount-display');

        if (discountRow) {
            discountRow.classList.remove('d-none');
        }
        if (discountAmt) {
            discountAmt.textContent = `−€${data.discount_amount.toFixed(2)}`;
        }
        if (totalEl) {
            totalEl.textContent = `€${data.new_total.toFixed(2)}`;
        }

        // checkout page – reload so the server computes the correct discounted total on the form
        const checkoutForm = document.getElementById('checkout-form');
        if (checkoutForm) {
            setTimeout(() => window.location.reload(), 800);
        }
    }

    // ── Event tracking ──────────────────────────────────────────────────────

    /**
     * Emit at most one widget interaction event (click XOR dismiss) per
     * decision — the simulator's event model samples exactly one outcome per
     * served widget, so repeat/duplicate interactions must not inflate reward.
     */
    function trackWidgetInteraction(eventType, decisionPoint, action, decisionId, propensity) {
        const key = String(decisionId);
        if (interactionSent[key]) return;
        interactionSent[key] = true;
        trackWidgetEvent(eventType, decisionPoint, action, decisionId, propensity);
    }

    function trackWidgetEvent(eventType, decisionPoint, action, decisionId, propensity) {
        fetch('/api/events', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: window.SESSION_ID,
                event_type: eventType,
                page: window.location.pathname,
                element: 'widget-slot',
                metadata: {
                    decision_id: decisionId,
                    decision_point: decisionPoint,
                    action: action,
                    propensity: propensity,
                },
            }),
            keepalive: true,
        }).catch(() => { });
    }

    // ── Scroll-depth 70% decision point ────────────────────────────────────

    function initScrollDecision(pageDepth) {
        let fired = false;
        const onScroll = () => {
            if (fired) return;
            const total = document.documentElement.scrollHeight;
            if (total <= window.innerHeight) return; // page fits entirely in viewport
            const ratio = (window.scrollY + window.innerHeight) / total;
            if (ratio < 0.70) return;
            fired = true;
            window.removeEventListener('scroll', onScroll);
            requestScrollDecision(pageDepth);
        };
        window.addEventListener('scroll', onScroll, { passive: true });
    }

    async function requestScrollDecision(pageDepth) {
        const context = {
            page_depth: pageDepth,
            referrer: getInitialReferrer(),
            screen_width: window.screen.width,
            screen_height: window.screen.height,
            scroll_pct: 70,
        };
        try {
            const res = await fetch(ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: window.SESSION_ID,
                    decision_point: 'scroll_engagement',
                    context,
                }),
            });
            if (!res.ok) {
                trackDecisionError('scroll_engagement', 'http_error', res.status);
                return;
            }
            const data = await res.json();
            if (data.action === 'no-op') return;
            if (data.action === 'help_popup') {
                showHelpPopup(data.decision_id, data.propensity);
                return;
            }
            // Every eligible action must render at the scroll point too —
            // otherwise the logged action differs from the shown intervention.
            await showScrollOverlayWidget(data.action, context, data.decision_id, data.propensity);
        } catch (_) {
            trackDecisionError('scroll_engagement', 'network_or_parse_error', 0);
        }
    }

    /** Render a non-help_popup action as a floating overlay after the scroll trigger. */
    async function showScrollOverlayWidget(action, context, decisionId, propensity) {
        const html = await renderWidget(action, context);
        if (!html) return;
        const overlay = document.createElement('div');
        overlay.id = 'scroll-widget-overlay';
        // .floating-widget keeps the overlay clear of the study-end card (see style.css).
        overlay.className = 'floating-widget';
        overlay.innerHTML = html;
        document.body.appendChild(overlay);

        trackWidgetEvent('widget_impression', 'scroll_engagement', action, decisionId, propensity);
        wireWidgetContainer(overlay, 'scroll_engagement', action, decisionId, propensity);
    }

    function showHelpPopup(decisionId, propensity) {
        const container = document.createElement('div');
        container.id = 'help-fab-container';
        container.innerHTML = `
            <button id="help-fab-btn" title="Need help?"
                    class="btn btn-primary rounded-circle shadow d-flex align-items-center justify-content-center floating-fab">
                <i class="bi bi-question-lg"></i>
            </button>
            <div id="help-popup-panel" class="card shadow floating-fab-panel" style="display:none;">
                <div class="card-header d-flex justify-content-between align-items-center py-2 px-3">
                    <span class="fw-semibold small"><i class="bi bi-headset me-1"></i>Need Help?</span>
                    <button id="help-popup-close" class="btn-close btn-sm" aria-label="Close"></button>
                </div>
                <div class="card-body p-2">
                    <div class="list-group list-group-flush">
                        <button class="list-group-item list-group-item-action small help-opt py-2">
                            <i class="bi bi-chat-dots me-2 text-primary"></i>Live Chat
                        </button>
                        <button class="list-group-item list-group-item-action small help-opt py-2">
                            <i class="bi bi-book me-2 text-success"></i>Browse FAQ
                        </button>
                        <button class="list-group-item list-group-item-action small help-opt py-2">
                            <i class="bi bi-envelope me-2 text-warning"></i>Contact Support
                        </button>
                    </div>
                </div>
            </div>`;

        document.body.appendChild(container);

        const fab = document.getElementById('help-fab-btn');
        const panel = document.getElementById('help-popup-panel');
        const close = document.getElementById('help-popup-close');

        trackWidgetEvent('widget_impression', 'scroll_engagement', 'help_popup', decisionId, propensity);

        // Opening the panel is the widget's click outcome (emitted at most once
        // per decision via trackWidgetInteraction).
        fab.addEventListener('click', () => {
            const isOpen = panel.style.display !== 'none';
            panel.style.display = isOpen ? 'none' : 'block';
            if (!isOpen) {
                trackWidgetInteraction('widget_click', 'scroll_engagement', 'help_popup', decisionId, propensity);
            }
        });

        close.addEventListener('click', () => {
            panel.style.display = 'none';
        });

        // Help option clicks share the same one-interaction-per-decision budget.
        container.querySelectorAll('.help-opt').forEach(btn => {
            btn.addEventListener('click', () => {
                trackWidgetInteraction('widget_click', 'scroll_engagement', 'help_popup', decisionId, propensity);
                const icon = btn.querySelector('i').outerHTML;
                btn.innerHTML = icon + ' ' + btn.textContent.trim() + ' <i class="bi bi-check-circle-fill text-success ms-1"></i>';
                btn.disabled = true;
            });
        });
    }
})();
