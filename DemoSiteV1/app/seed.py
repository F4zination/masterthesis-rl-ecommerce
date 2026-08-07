from .database import SessionLocal, engine, Base
from .models import Category, Product


def slugify(text: str) -> str:
    return text.lower().replace(" ", "-").replace("&", "and").replace("'", "")


CATEGORIES = [
    {"name": "Electronics", "description": "Gadgets, peripherals, and tech accessories"},
    {"name": "Clothing", "description": "Apparel for every occasion"},
    {"name": "Books", "description": "Fiction, non-fiction, and technical reads"},
    {"name": "Home & Kitchen", "description": "Everything for your living space"},
    {"name": "Sports", "description": "Gear and equipment for an active lifestyle"},
    {"name": "Accessories", "description": "Watches, wallets, bags, and more"},
]

PRODUCTS = {
    "Electronics": [
        ("Wireless Headphones", "Premium noise-cancelling over-ear headphones with 30h battery life.", 89.99),
        ("Mechanical Keyboard", "RGB mechanical keyboard with Cherry MX switches.", 129.99),
        ("27-inch Monitor", "4K IPS monitor with USB-C connectivity.", 349.99),
        ("Bluetooth Speaker", "Portable waterproof speaker with 360° sound.", 49.99),
        ("USB-C Hub", "7-in-1 USB-C hub with HDMI, SD card, and ethernet.", 39.99),
        ("Webcam HD", "1080p webcam with built-in microphone and privacy cover.", 59.99),
        ("Wireless Mouse", "Ergonomic wireless mouse with silent clicks.", 29.99),
        ("Portable SSD 1TB", "Fast external SSD with USB 3.2 interface.", 79.99),
        ("Smart LED Strip", "WiFi-enabled RGB LED strip, 5 meters, app-controlled.", 24.99),
        ("Laptop Stand", "Adjustable aluminum laptop stand with cooling ventilation.", 34.99),
    ],
    "Clothing": [
        ("Classic White T-Shirt", "100% organic cotton crew-neck tee.", 19.99),
        ("Slim Fit Jeans", "Dark wash stretch denim with modern slim fit.", 49.99),
        ("Hooded Sweatshirt", "Cozy fleece-lined hoodie with kangaroo pocket.", 39.99),
        ("Rain Jacket", "Lightweight waterproof jacket with sealed seams.", 79.99),
        ("Wool Beanie", "Soft merino wool beanie in charcoal grey.", 14.99),
        ("Linen Shirt", "Breathable linen button-down for warm days.", 44.99),
        ("Running Shorts", "Quick-dry athletic shorts with zippered pocket.", 29.99),
        ("Winter Parka", "Insulated parka with faux-fur hood trim.", 159.99),
        ("Casual Sneakers", "Minimalist canvas sneakers with rubber sole.", 54.99),
        ("Cotton Socks 5-Pack", "Everyday ankle socks in assorted colors.", 12.99),
    ],
    "Books": [
        ("The Pragmatic Programmer", "Classic software development handbook by Hunt & Thomas.", 42.99),
        ("Deep Learning with Python", "Hands-on introduction to deep learning by François Chollet.", 49.99),
        ("Dune", "Epic science fiction novel by Frank Herbert.", 12.99),
        ("Sapiens", "A brief history of humankind by Yuval Noah Harari.", 15.99),
        ("Clean Code", "A handbook of agile software craftsmanship by Robert C. Martin.", 37.99),
        ("The Great Gatsby", "Timeless American classic by F. Scott Fitzgerald.", 9.99),
        ("Reinforcement Learning: An Introduction", "The definitive RL textbook by Sutton & Barto.", 54.99),
        ("Atomic Habits", "Practical strategies for building good habits by James Clear.", 16.99),
    ],
    "Home & Kitchen": [
        ("Ceramic Coffee Mug", "Handcrafted 350ml mug with minimalist design.", 14.99),
        ("Desk Lamp LED", "Dimmable LED desk lamp with USB charging port.", 39.99),
        ("Cast Iron Skillet", "Pre-seasoned 12-inch cast iron pan.", 34.99),
        ("Bamboo Cutting Board", "Large antibacterial bamboo board with juice groove.", 22.99),
        ("French Press Coffee Maker", "Borosilicate glass french press, 1 liter.", 29.99),
        ("Scented Candle Set", "Set of 3 soy wax candles: vanilla, lavender, cedar.", 24.99),
        ("Kitchen Timer", "Magnetic digital timer with loud alarm.", 9.99),
        ("Plant Pot Set", "Set of 3 ceramic plant pots with drainage holes.", 27.99),
    ],
    "Sports": [
        ("Yoga Mat", "Non-slip TPE yoga mat, 6mm thick, with carrying strap.", 29.99),
        ("Stainless Water Bottle", "Double-walled insulated bottle, keeps drinks cold 24h.", 19.99),
        ("Resistance Bands Set", "Set of 5 latex bands with varying resistance levels.", 16.99),
        ("Jump Rope", "Speed jump rope with ball-bearing handles.", 12.99),
        ("Foam Roller", "High-density foam roller for muscle recovery, 45cm.", 24.99),
        ("Dumbbell Set 10kg", "Pair of neoprene-coated dumbbells, 5kg each.", 39.99),
        ("Sports Towel", "Quick-dry microfiber towel with zip pocket.", 14.99),
    ],
    "Accessories": [
        ("Minimalist Watch", "Stainless steel case with leather strap, Japanese movement.", 89.99),
        ("Leather Wallet", "Slim bifold wallet with RFID blocking.", 34.99),
        ("Canvas Backpack", "30L daily backpack with laptop compartment.", 59.99),
        ("Sunglasses Polarized", "UV400 polarized lenses with acetate frame.", 44.99),
        ("Phone Case", "Shock-absorbing clear case with MagSafe compatibility.", 19.99),
        ("Keychain Multi-tool", "Compact 10-in-1 stainless steel multi-tool.", 14.99),
        ("Belt Leather", "Full-grain leather belt with brushed nickel buckle.", 29.99),
    ],
}


def seed_database():
    """Populate the database with categories and products if empty."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(Category).count() > 0:
            return False  # Already seeded

        categories = {}
        for cat_data in CATEGORIES:
            cat = Category(
                name=cat_data["name"],
                slug=slugify(cat_data["name"]),
                description=cat_data["description"],
            )
            db.add(cat)
            db.flush()
            categories[cat_data["name"]] = cat.id

        product_id = 0
        for cat_name, products in PRODUCTS.items():
            for name, description, price in products:
                product_id += 1
                product = Product(
                    name=name,
                    slug=slugify(name),
                    description=description,
                    price=price,
                    image_url=f"https://picsum.photos/seed/{product_id}/400/400",
                    category_id=categories[cat_name],
                    stock=100,
                )
                db.add(product)

        db.commit()
        return True
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
    print("Database seeded successfully.")
