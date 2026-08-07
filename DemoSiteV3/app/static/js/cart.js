/**
 * Cart interactions – Add, update quantity, remove via /api/cart/*
 */
(function () {
    // --- Add to Cart ---
    document.addEventListener('click', async (e) => {
        const btn = e.target.closest('.add-to-cart-btn');
        if (!btn) return;

        const productId = parseInt(btn.dataset.productId, 10);
        btn.disabled = true;
        btn.innerHTML = '<i class="bi bi-check"></i> Added';

        try {
            const res = await fetch('/api/cart/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ product_id: productId, quantity: 1 }),
            });
            const data = await res.json();
            updateBadge(data.count);

            // Track add_to_cart event
            fetch('/api/events', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: window.SESSION_ID,
                    event_type: 'add_to_cart',
                    page: window.location.pathname,
                    metadata: { product_id: productId },
                }),
                keepalive: true,
            }).catch(() => { });
        } catch (_) { }

        setTimeout(() => {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-cart-plus"></i> Add to Cart';
        }, 1200);
    });

    // --- Quantity Buttons (cart page) ---
    document.addEventListener('click', async (e) => {
        const btn = e.target.closest('.cart-qty-btn');
        if (!btn) return;

        const productId = parseInt(btn.dataset.productId, 10);
        const delta = parseInt(btn.dataset.delta, 10);
        const row = btn.closest('tr');
        const input = row.querySelector('.cart-qty-input');
        const newQty = parseInt(input.value, 10) + delta;

        if (newQty <= 0) {
            await removeItem(productId, row);
            return;
        }

        try {
            const res = await fetch('/api/cart/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ product_id: productId, quantity: newQty }),
            });
            const data = await res.json();
            input.value = newQty;
            updateBadge(data.count);
            refreshCartPage();
        } catch (_) { }
    });

    // --- Remove Button (cart page) ---
    document.addEventListener('click', async (e) => {
        const btn = e.target.closest('.cart-remove-btn');
        if (!btn) return;
        const productId = parseInt(btn.dataset.productId, 10);
        const row = btn.closest('tr');
        await removeItem(productId, row);
    });

    async function removeItem(productId, row) {
        try {
            const res = await fetch('/api/cart/remove', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ product_id: productId }),
            });
            const data = await res.json();
            if (row) row.remove();
            updateBadge(data.count);
            refreshCartPage();

            // Track remove event
            fetch('/api/events', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: window.SESSION_ID,
                    event_type: 'remove_from_cart',
                    page: window.location.pathname,
                    metadata: { product_id: productId },
                }),
                keepalive: true,
            }).catch(() => { });
        } catch (_) { }
    }

    function updateBadge(count) {
        const badge = document.getElementById('cart-badge');
        if (!badge) return;
        badge.textContent = count;
        badge.classList.toggle('d-none', count === 0);
    }

    async function refreshCartPage() {
        // Only refresh totals on cart page
        const totalEl = document.getElementById('cart-total');
        if (!totalEl) return;
        try {
            const res = await fetch('/api/cart');
            const data = await res.json();
            totalEl.textContent = `€${data.total.toFixed(2)}`;
            if (data.items.length === 0) {
                window.location.reload();
            }
        } catch (_) { }
    }
})();
