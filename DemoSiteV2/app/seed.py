import re
import sys

from .database import SessionLocal, engine, Base
from .models import Category, Product, CartItem, OrderItem, Order


def slugify(text: str) -> str:
    return re.sub(r'[^\w-]', '', text.lower().replace(" ", "-").replace("&", "and").replace("'", ""))



def _local(filename: str) -> str:
    """Return the static URL for a custom product image.

    Place the image file in ``app/static/images/products/`` and pass its
    filename here (e.g. ``_local("my-headphones.png")``).  Use this instead
    of ``_lf(...)`` for any product you want to supply a real photo for.

    Args:
        filename: The image filename including extension.

    Returns:
        A root-relative URL served by FastAPI's static file mount.
    """
    return f"/static/images/products/{filename}"




CATEGORIES = [
    {"name": "Electronics", "description": "Gadgets, peripherals, and tech accessories"},
    {"name": "Clothing",    "description": "Apparel for every occasion"},
    {"name": "Home & Kitchen", "description": "Everything for your living space"},
    {"name": "Sports",      "description": "Gear and equipment for an active lifestyle"},
    {"name": "Accessories", "description": "Watches, wallets, bags, and more"},
]

# Each product: (name, description, price, image_url)
PRODUCTS = {
    # ── Electronics ──────────────────────────────────────────────────────────
    "Electronics": [
        ("Wireless Headphones",         "Premium noise-cancelling over-ear headphones with 30h battery life.", 89.99,  _local("electronics/WirelessHeadphones.png")),
        ("Mechanical Keyboard",         "RGB mechanical keyboard with Cherry MX Brown switches.", 129.99, _local("electronics/MechanicalKeyboard.png")),
        ("27-inch 4K Monitor",          "4K IPS display with USB-C, HDR400, and 144Hz refresh rate.", 349.99, _local("electronics/27-inch4kMonitor.png")),
        ("Bluetooth Speaker",           "Portable waterproof speaker with 360° sound and 20h battery.", 49.99,  _local("electronics/BluetoothSpeaker.png")),
        ("USB-C Hub 7-in-1",            "HDMI 4K, 3xUSB-A, SD/microSD, 100W PD pass-through.", 39.99,  _local("electronics/USB-C_Hub.png")),
        ("Webcam 1080p HD",             "Full HD webcam with auto-focus, built-in noise-cancelling mic.", 59.99,  _local("electronics/Webcam1080p.png")),
        ("Wireless Mouse Ergonomic",    "Sculpted ergonomic wireless mouse, 70-day battery, silent clicks.", 29.99,  _local("electronics/ErgonomicMouse.png")),
        ("Portable SSD 1TB",            "USB 3.2 Gen 2 external SSD — up to 1050 MB/s read speed.", 79.99,  _local("electronics/Nvme500GB.png")),
        ("Smart LED Strip 5m",          "Wi-Fi RGB+W strip, voice-control ready, music-sync mode.", 24.99,  _local("electronics/LedStrip.png")),
        ("Laptop Stand Aluminum",       "Height-adjustable aluminium stand with ventilation slots.", 34.99,  _local("electronics/StandAluminium.png")),
        ("Gaming Headset 7.1",          "Surround-sound gaming headset with detachable boom mic.", 69.99,  _local("electronics/WirelessHeadphones.png")),
        ("Wireless Charger 15W",        "Qi2 fast-wireless charging pad, compatible with all Qi devices.", 29.99, _local("electronics/WirelessCharger.png")),
        ("Smart Plug 4-Pack",           "Wi-Fi smart plugs with energy monitoring and scheduling app.", 34.99, _local("electronics/4PackPlugs.png")),
        ("True Wireless Earbuds",       "ANC earbuds, 8h playtime + 24h case, IPX5 water resistance.", 119.99, _local("electronics/WirelessEarBuds.png")),
        ("Mini PC Stick",               "Quad-core mini PC stick, 8GB RAM, 128GB storage, HDMI.", 179.99, _local("electronics/PCStick.png")),
        ("Drawing Tablet A5",           "Pen tablet with 8192-level pressure, tilt recognition.", 99.99,  _local("electronics/DrawingTables.png")),
        ("4K HDMI Cable 2m",            "Ultra-high-speed HDMI 2.1 cable, 4K@120Hz, 8K@60Hz.", 14.99,  _local("electronics/UsbC-Cable.png")),
        ("Privacy Screen Filter 15.6",  "Anti-spy matte filter for 16:9 laptops, easy-attach tabs.", 29.99,  _local("electronics/PrivacyScreenFilter.png")),
        ("Laptop Sleeve 15 inch",       "Neoprene laptop sleeve with accessory pocket, fits up to 15.6\".", 19.99, _local("electronics/LaptopBag.png")),
        ("USB Microphone",              "Cardioid condenser USB mic with gain control and headphone out.", 79.99,  _local("electronics/DeskMicrophone.png")),
        ("Streaming Webcam 4K",         "4K autofocus webcam with software-based background removal.", 149.99, _local("electronics/StreamingWebcam.png")),
        ("Compact Numpad",              "Wireless numpad with backlight, rechargeable battery.", 44.99,  _local("electronics/NumPad.png")),
        ("Vertical Ergonomic Mouse",    "57° vertical grip reduces wrist strain; DPI: 800-4000.", 39.99,  _local("electronics/VerticalErgonomicMouse.png")),
        ("Cable Management Box",        "Desktop cable organiser box with 5 cable slots, lid included.", 17.99, _local("electronics/CableManagementBox.png")),
        ("USB-C Cables 3-Pack",         "Braided USB-C cables (0.5m, 1m, 2m), 60W fast-charge.", 19.99,  _local("electronics/UsbC-Cable.png")),
        ("Power Bank 20000mAh",         "Dual USB-A + USB-C output, 22.5W fast-charge, LED indicator.", 49.99, _local("electronics/Powerbank.png")),
        ("Adjustable Phone Stand",      "Foldable aluminium stand, adjustable angle, for desk or bed.", 12.99, _local("electronics/PhoneStand.png")),
        ("Smart Home Hub",              "Zigbee/Z-Wave hub, controls up to 100 smart devices.", 69.99,  _local("electronics/SmartHomeHubh.png")),
        ("Wi-Fi 6 Range Extender",      "AX1800 dual-band extender, mesh-compatible, Gigabit port.", 54.99, _local("electronics/WifiRangeExtender.png")),
        ("Thunderbolt 4 Dock",          "11-port Thunderbolt 4 dock: 4K dual display, 96W charging.", 169.99, _local("electronics/Thunderbold4Dock.png")),
        ("Memory Foam Wrist Rest",      "Slow-rebound wrist rest for keyboard, non-slip base.", 17.99,  _local("electronics/MemoryFoamWristRest.png")),
        ("Single Monitor Arm",          "Gas-spring monitor arm, VESA 75/100, cable management channel.", 49.99, _local("electronics/SingleMontiorArm.png")),
        ("XL Wireless Desk Pad",        "900x400mm desk pad with built-in wireless charging zone.", 54.99,  _local("electronics/XL-WirelessDeskPad.png")),
        ("Portable Mini Projector",     "1080p native, 300 ANSI lumens, battery-powered, built-in speaker.", 199.99, _local("electronics/MiniProjector.png")),
        ("Mechanical Switch Tester",    "72-switch RGB tester kit with keycaps; compare every popular switch.", 22.99, _local("electronics/SwitchTestStand.png")),
        ("RGB Mouse Pad XL",            "Extended RGB gaming pad, 900x400mm, anti-fray stitching.", 24.99,  _local("electronics/RGBMousePad.png")),
        ("4K HDMI Switch 3-Port",       "Auto-switching 3-in-1 HDMI 2.0 switch with remote control.", 29.99, _local("electronics/3PortHDMISwitch.png")),
        ("USB 3.2 Card Reader",         "SD, microSD, CF, and CFexpress reader in one compact hub.", 19.99,  _local("electronics/SDCardReader.png")),
        ("Wireless Number Pad",         "Slim Bluetooth numpad with backlight, multi-device pairing.", 39.99, _local("electronics/NumPad.png")),
        ("Smart Display Speaker",       "7-inch touchscreen smart speaker with voice assistant built-in.", 89.99, _local("electronics/SmartDisplaySpeaker.png")),
        ("8-Port Network Switch",       "Unmanaged Gigabit switch, plug-and-play, fanless design.", 34.99,  _local("electronics/NetworkSwitch.png")),
        ("Digital Soldering Station",   "65W digital soldering station with PID temperature control.", 59.99,  _local("electronics/DigitalSolderingStation.png")),
        ("Auto Digital Multimeter",     "Auto-ranging multimeter: AC/DC voltage, current, continuity.", 24.99, _local("electronics/MultiMeter.png")),
        ("Screen Cleaning Kit",         "3-piece kit: microfibre cloth, brush, and spray bottle.", 9.99,   _local("electronics/ScreenCleaningKit.png")),
        ("120mm PC Fan 3-Pack",         "Addressable RGB 120mm fans, PWM, magnetic levitation bearing.", 29.99, _local("electronics/120mmRGBFans.png")),
        ("Laptop Cooling Pad",          "5-fan cooling pad with adjustable stand and dual USB passthrough.", 27.99, _local("electronics/LaptopCoolingPad.png")),
        ("Anti-Static Wrist Strap",     "Adjustable ESD wrist strap with 1MΩ resistor and coiled cord.", 7.99,  _local("electronics/AntiStaticWristBand.png")),
        ("NVMe SSD 500GB M.2",          "PCIe 4.0 NVMe SSD — 7000 MB/s read, 6500 MB/s write.", 69.99,  _local("electronics/Nvme500GB.png")),
        ("DDR5 RAM 16GB Kit",           "16GB (2x8GB) DDR5-5200 CL38 desktop memory kit.", 99.99,  _local("electronics/DDR5RAM.png")),
        ("120mm AIO CPU Cooler",        "All-in-one liquid cooler, 120mm radiator, ARGB pump head.", 64.99,  _local("electronics/AIO_CPU-Cooler.png")),
    ],

    # ── Clothing ─────────────────────────────────────────────────────────────
    "Clothing": [
        ("Classic White T-Shirt",      "100% organic cotton crew-neck tee, pre-shrunk, relaxed fit.", 19.99, _local("clothing/WhiteTShirt.png")),
        ("Slim Fit Jeans",             "Dark-wash stretch denim, slim fit, double-stitched seams.", 49.99,  _local("clothing/FitJeans.png")),
        ("Hooded Sweatshirt",          "280gsm brushed-fleece hoodie with kangaroo pocket, unisex.", 39.99, _local("clothing/Hoodie.png")),
        ("Rain Jacket",                "Lightweight waterproof jacket with taped seams and mesh lining.", 79.99, _local("clothing/RainJacket.png")),
        ("Wool Beanie",                "100% merino wool beanie, ribbed knit, one size fits all.", 14.99,  _local("clothing/Beanie.png")),
        ("Linen Shirt",                "Breathable linen button-down, relaxed cut, summer-ready.", 44.99,  _local("clothing/LinenShirt.png")),
        ("Running Shorts",             "Quick-dry 5\" shorts with inner brief and zippered pocket.", 29.99, _local("clothing/RunningShorts.png")),
        ("Winter Parka",               "600-fill down parka with coyote-fur-trim hood, waterproof shell.", 159.99, _local("clothing/WinterParka.png")),
        ("Casual Sneakers",            "Low-top canvas sneakers with vulcanised rubber sole.", 54.99,  _local("clothing/CasualSneeker.png")),
        ("Cotton Socks 5-Pack",        "Reinforced-heel crew socks in 5 classic colours.", 12.99,  _local("clothing/5SockPack.png")),
        ("Graphic Print T-Shirt",      "Heavyweight 200gsm tee with front screen-print graphic.", 22.99, _local("clothing/GraphicPrintShirt.png")),
        ("Chino Pants",                "Slim-fit stretch chinos, wrinkle-resistant, flat-front.", 54.99,  _local("clothing/ChinoPants.png")),
        ("Polo Shirt",                 "Piqué cotton polo with two-button placket and ribbed collar.", 34.99, _local("clothing/PoloShirt.png")),
        ("Denim Jacket",               "Washed denim trucker jacket with adjustable waist tabs.", 69.99,  _local("clothing/JeansJacket.png")),
        ("Fleece Vest",                "Anti-pill fleece vest with full YKK zip and two pockets.", 49.99,  _local("clothing/FleeceVest.png")),
        ("Oxford Button-Down Shirt",   "Classic OCBD shirt, non-iron finish, slim fit.", 44.99,  _local("clothing/OxfortButtonDownShirt.png")),
        ("Compression Leggings",       "7/8-length high-waist leggings with 4-way stretch fabric.", 34.99, _local("clothing/Leggins.png")),
        ("Windbreaker Jacket",         "Packable windbreaker with reflective trim, stuff-sack included.", 89.99, _local("clothing/WindbreakerJacket.png")),
        ("Chelsea Boots",              "Leather-upper Chelsea boots with elasticated side panels.", 99.99,  _local("clothing/ChelseaBoots.png")),
        ("Merino Wool Sweater",        "Midweight 100% merino crewneck, itch-free, temperature-regulating.", 79.99, _local("clothing/WoolSweater.png")),
        ("Baseball Cap",               "6-panel structured cap with curved brim and back strap.", 19.99,  _local("clothing/BaseballCap.png")),
        ("Cycling Jersey",             "Race-fit cycling jersey with 3 back pockets, UPF 50.", 64.99,  _local("clothing/CyclinJersey.png")),
        ("Thermal Base Layer Set",     "Long-sleeve top + bottoms moisture-wicking base layer set.", 44.99, _local("clothing/ThermalBaseLayerSet.png")),
        ("Board Shorts",               "Quick-dry 18\" board shorts with mesh liner and velcro fly.", 34.99, _local("clothing/BoardShorts.png")),
        ("Cargo Pants",                "Relaxed-fit cargo pants with 6 pockets, cotton-ripstop blend.", 59.99, _local("clothing/CargoPants.png")),
        ("V-Neck Sweater",             "Fine-knit merino V-neck, ribbed cuffs and hem.", 39.99,  _local("clothing/VNeckSweater.png")),
        ("Zip-Up Hoodie",              "Full-zip fleece hoodie with thumb holes and media pocket.", 54.99, _local("clothing/ZipUpHoodie.png")),
        ("Trail Running Shoes",        "Grippy trail shoes with rock-plate protection, waterproof upper.", 89.99, _local("clothing/TrailRunningShoes.png")),
        ("Swim Trunks",                "Ultra-light 14\" swim trunks with quick-dry mesh lining.", 24.99, _local("clothing/SwimTrunk.png")),
        ("Long Sleeve Henley",         "Garment-dyed henley in heavy cotton, 3-button placket.", 34.99, _local("clothing/LongSleveHenry.png")),
        ("Athletic Tank Top",          "Mesh-panel tank with moisture-wicking tech, racerback cut.", 19.99, _local("clothing/AthleticTankTop.png")),
        ("Flannel Shirt",              "Soft brushed flannel in plaid check, button-down front.", 49.99, _local("clothing/FlannelShirt.png")),
        ("Yoga Pants",                 "High-rise yoga pants with hidden waistband pocket.", 44.99,  _local("clothing/YogaPants.png")),
        ("Puffer Jacket",              "Lightweight recycled-fill puffer, packable, water-repellent.", 119.99, _local("clothing/PufferJacket.png")),
        ("Loafers",                    "Suede penny loafers with leather insole and rubber outsole.", 84.99,  _local("clothing/Loafers.png")),
        ("Casual Dress Shirt",         "Slim-fit poplin shirt, wrinkle-resistant, spread collar.", 39.99, _local("clothing/CasualDressShirt.png")),
        ("Basketball Shorts",          "Loose-fit mesh shorts with 11\" inseam, elastic waistband.", 29.99, _local("clothing/BaseballShorts.png")),
        ("Quarter-Zip Pullover",       "Midlayer fleece quarter-zip with stand-up collar.", 59.99,  _local("clothing/QuarterZipPullover.png")),
        ("Sandals",                    "Contoured-footbed sandals with adjustable hook-and-loop straps.", 34.99, _local("clothing/Sandals.png")),
        ("Knit Scarf",                 "Chunky-knit scarf in 100% lambswool, 180cm length.", 19.99,  _local("clothing/KnitScarf.png")),
        ("Waterproof Hiking Boots",    "GORE-TEX hiking boots with Vibram sole and ankle support.", 129.99, _local("clothing/WaterproofHikingBoots.png")),
        ("Sleeveless Workout Top",     "Oversized slub-cotton vest, dropped armholes.", 24.99,  _local("clothing/SleevelessWorkoutTop.png")),
        ("Slim Fit Chinos",            "Slim-tapered chinos in stretch twill, no-iron finish.", 54.99,  _local("clothing/SlimFitChinos.png")),
        ("Crewneck Sweatshirt",        "350gsm French-terry crewneck, dropped shoulder fit.", 39.99, _local("clothing/CrewneckSweatshirt.png")),
        ("High-Waist Leggings",        "Squat-proof high-waist leggings with side pocket.", 44.99,  _local("clothing/HighWaistLeggins.png")),
        ("Formal Blazer",              "Slim-fit single-breasted blazer in stretch wool blend.", 149.99, _local("clothing/FormalBlazer.png")),
        ("Hip Pack",                   "600D ripstop hip pack with two zipped compartments, 1.5L.", 29.99, _local("clothing/HipPack.png")),
        ("Genuine Leather Belt",       "Full-grain leather belt, 35mm width, brushed-silver buckle.", 27.99, _local("clothing/LeatherBelt.png")),
        ("Sun Hat",                    "Wide-brim UPF 50+ sun hat, crushable, with chin cord.", 24.99,  _local("clothing/SunHat.png")),
        ("Performance Polo",           "Moisture-wicking polo with UV protection, slim fit.", 44.99,  _local("clothing/PerformancePolo.png")),
    ],

    # ── Home & Kitchen ────────────────────────────────────────────────────────
    "Home & Kitchen": [
        ("Ceramic Coffee Mug",            "Hand-thrown 350ml stoneware mug with matte glaze.", 14.99, _local("home&kitchen/CeramicCoffeeMug.png")),
        ("Desk Lamp LED",                 "Dimmable LED lamp, 5 colour temps, USB-A charging port.", 39.99, _local("home&kitchen/DeskLampLED.png")),
        ("Cast Iron Skillet 12 inch",     "Pre-seasoned cast iron pan, works on all hob types.", 34.99, _local("home&kitchen/CastIronSkillet.png")),
        ("Bamboo Cutting Board",          "Large antibacterial bamboo board with deep juice groove.", 22.99, _local("home&kitchen/BambooCuttingBoard.png")),
        ("French Press Coffee Maker",     "Borosilicate glass french press, 1-litre, stainless plunger.", 29.99, _local("home&kitchen/FrenchPress.png")),
        ("Scented Candle Set",            "3 soy-wax candles: vanilla, lavender, and cedar — 50h each.", 24.99, _local("home&kitchen/ScentedCandleSet.png")),
        ("Magnetic Kitchen Timer",        "Strong-magnet digital timer, large display, loud alarm.", 9.99,  _local("home&kitchen/MagneticKitchenTimer.png")),
        ("Ceramic Plant Pot Set",         "Set of 3 minimalist pots with drainage holes and saucers.", 27.99, _local("home&kitchen/CeramicPlantPotSet.png")),
        ("Pour-Over Coffee Set",          "Borosilicate dripper, server, and 100 paper filters.", 39.99, _local("home&kitchen/Pour-OverCoffeeSEt.png")),
        ("Glass Water Pitcher 1.5L",      "Borosilicate glass pitcher with silicone handle, dishwasher-safe.", 19.99, _local("home&kitchen/GlassWaterPitcher.png")),
        ("Airtight Container Set 7-Piece","BPA-free airtight food storage containers, various sizes.", 29.99, _local("home&kitchen/AirtightContainerSet.png")),
        ("Silicone Spatula Set",          "Heat-resistant silicone spatulas (3 sizes), one-piece moulded.", 14.99, _local("home&kitchen/SiliconeSpatulaSet.png")),
        ("Stainless Mixing Bowls 3-Pack", "Graduated stainless steel bowls with non-slip base.", 24.99, _local("home&kitchen/StainlessMixingBowls.png")),
        ("Non-Stick Frying Pan 28cm",     "PFOA-free granite-coated frying pan, induction compatible.", 34.99, _local("home&kitchen/Non-StickFryingPan.png")),
        ("Electric Kettle 1.7L",          "Temperature-select kettle: 5 presets, keep-warm function.", 44.99, _local("home&kitchen/ElectricKettle.png")),
        ("Knife Block Set 6-Piece",       "German high-carbon stainless blades with full-tang walnut handles.", 79.99, _local("home&kitchen/KnifeBlockSet.png")),
        ("Beeswax Wrap 3-Pack",           "Reusable beeswax food wraps — S, M, L — zero plastic.", 14.99, _local("home&kitchen/BeewaxWrap.png")),
        ("Burr Coffee Grinder",           "Conical burr grinder, 15 grind settings, 200g hopper.", 49.99, _local("home&kitchen/BurrCoffeeGrinder.png")),
        ("Dish Drying Rack",              "Stainless rack with removable drip tray and utensil holder.", 27.99, _local("home&kitchen/DishDryingRack.png")),
        ("Vacuum Storage Bags 8-Pack",    "Hand-pump vacuum bags — 2xjumbo, 3xlarge, 3xmedium.", 19.99, _local("home&kitchen/VaccumStorageBags.png")),
        ("Ceramic Soap Dispenser",        "Matte ceramic pump dispenser, 300ml, for countertop use.", 17.99, _local("home&kitchen/CeramicSoapDispenser.png")),
        ("Linen Dish Towels 4-Pack",      "100% stonewashed linen towels with hanging loop.", 19.99, _local("home&kitchen/DishTowels.png")),
        ("Instant-Read Thermometer",      "Foldable probe thermometer, reads in 2 seconds, ±0.5°C.", 24.99, _local("home&kitchen/InstantReadThermometer.png")),
        ("Espresso Tamper 58mm",          "Stainless flat-base tamper, calibrated spring at 15kg.", 19.99, _local("home&kitchen/EspressoTamper.png")),
        ("Reusable Produce Bags 10-Pack", "Washable mesh bags for fruit and vegetables — zero waste.", 12.99, _local("home&kitchen/ReusableProduceBags.png")),
        ("Stainless Kitchen Shears",      "Heavy-duty shears that split into two for easy cleaning.", 14.99, _local("home&kitchen/KitchenScissors.png")),
        ("Rotating Spice Rack",           "2-tier rotating spice organiser, holds 24 standard jars.", 29.99, _local("home&kitchen/RotatingSpiceRack.png")),
        ("Silicone Oven Mitts Pair",      "Extra-long 43cm mitts, heat-resistant to 250°C.", 17.99, _local("home&kitchen/OvenGlovers.png")),
        ("Copper Measuring Cups Set",     "5-piece stainless cups with rose-gold finish and engraved measures.", 22.99, _local("home&kitchen/CopperMeasuringCups.png")),
        ("Countertop Compost Bin",        "1.3-litre stainless bin with charcoal filter to prevent odours.", 24.99, _local("home&kitchen/TableCompost.png")),
        ("Marble Coaster Set 4-Pack",     "Genuine marble coasters with cork backing, 10cm diameter.", 19.99, _local("home&kitchen/MarbleCoasterSet.png")),
        ("Glass Oil Dispenser 500ml",     "Borosilicate glass oil bottle with no-drip pouring spout.", 14.99, _local("home&kitchen/GlassOilDispenser.png")),
        ("Glass Meal Prep Containers 5-Pack", "Oven-safe glass containers with snap-lock lids, various sizes.", 34.99, _local("home&kitchen/GlassMealPrepContainers.png")),
        ("Wooden Spoon Set",              "Set of 5 long-handle beechwood cooking spoons.", 12.99, _local("home&kitchen/WoodenSpoonSet.png")),
        ("Handheld Milk Frother",         "Battery-powered frother — whisks milk in under 20 seconds.", 9.99,  _local("home&kitchen/HandheldMilkFrother.png")),
        ("Fruit Basket Wire",             "Powder-coated wire fruit bowl with removable two-tier design.", 19.99, _local("home&kitchen/FruitBasketWire.png")),
        ("Reusable Silicone Zip Bags",    "Freezer-safe silicone zip bags (2xsandwich, 2xsnack).", 22.99, _local("home&kitchen/ReusableSiliconeZipBags.png")),
        ("Magnetic Knife Strip 40cm",     "Powder-coated steel knife strip with concealed wall fixings.", 27.99, _local("home&kitchen/MagneticKnifeStrip.png")),
        ("Marble Mortar and Pestle",      "Solid marble mortar and pestle, 15cm diameter.", 24.99, _local("home&kitchen/MarbleMortar.png")),
        ("Digital Kitchen Scale",         "Precision scale to 5kg in 1g increments, tare function.", 19.99, _local("home&kitchen/DigitalKitchenScale.png")),
        ("Pastry Brush Set",              "Set of 3 silicone pastry brushes — spread without shed.", 9.99,  _local("home&kitchen/PastryBrushSet.png")),
        ("Stainless Bread Box",           "Large roll-top bread bin, keeps loaves fresh up to 5 days.", 34.99, _local("home&kitchen/StainlessBreadBox.png")),
        ("Large Salad Spinner",           "5-litre pull-cord salad spinner with non-slip base.", 27.99, _local("home&kitchen/LargeSaladSpinner.png")),
        ("Bottle Brush Set",              "5-piece nylon bottle-brush set for cups, straws, and bottles.", 12.99, _local("home&kitchen/BottleBrushSet.png")),
        ("Collapsible Silicone Colander", "Foldable colander that stores flat, dishwasher-safe.", 17.99, _local("home&kitchen/CollapsibleSlilconeColander.png")),
        ("Avocado Slicer 3-in-1",         "Split, pit, and slice avocados safely with one tool.", 8.99,  _local("home&kitchen/AvocadoCutter3-1.png")),
        ("Herb Keeper",                   "Tall refrigerator herb keeper with water reservoir, 28cm.", 14.99, _local("home&kitchen/HerbKeepter.png")),
        ("Wooden Cutting Board Set",      "Set of 3 graduated acacia boards with juice groove.", 39.99, _local("home&kitchen/WoodenCuttingBoardSet.png")),
        ("Handmade Pottery Bowl",         "Large hand-painted ceramic serving bowl, 28cm, each unique.", 29.99, _local("home&kitchen/HandmadePotteryBowl.png")),
        ("Insulated Lunch Box",           "Double-walled stainless lunchbox, keeps hot 6h / cold 8h.", 27.99, _local("home&kitchen/InsulatedLunchBox.png")),
    ],

    # ── Sports ────────────────────────────────────────────────────────────────
    "Sports": [
        ("Yoga Mat Non-Slip",              "6mm TPE yoga mat with alignment lines and carry strap.", 29.99, _local("sports/YogaMat.png")),
        ("Stainless Water Bottle 750ml",   "Double-walled vacuum bottle — 24h cold, 12h hot.", 19.99, _local("sports/StainlessWaterBottle.png")),
        ("Resistance Bands Set 5-Pack",    "Latex loop bands in 5 resistances (5-25kg) with carry bag.", 16.99, _local("sports/ResistanceBand.png")),
        ("Speed Jump Rope",                "4mm cable jump rope with ball-bearing handles, adjustable length.", 12.99, _local("sports/SpeedJumpRope.png")),
        ("Foam Roller 45cm",               "High-density EPP foam roller with textured surface.", 24.99, _local("sports/FoamRoller.png")),
        ("Dumbbell Set 10kg",              "Pair of neoprene-coated hex dumbbells, 5kg each.", 39.99, _local("sports/DumbbellSet10kg.png")),
        ("Quick-Dry Sports Towel",         "Microfibre gym towel with zip side pocket, XL 100x50cm.", 14.99, _local("sports/SportTowel.png")),
        ("Weightlifting Gloves",           "Ventilated half-finger gloves with wrist-wrap support.", 19.99, _local("sports/WeightliftingGloves.png")),
        ("Adjustable Kettlebell 16kg",     "Cast-iron kettlebell with vinyl coating, flat base.", 79.99, _local("sports/Kettlebell.png")),
        ("Doorframe Pull-Up Bar",          "No-drill doorframe bar, supports up to 150kg.", 34.99, _local("sports/DoorframePullUpBar.png")),
        ("Digital Jump Rope Counter",      "Jump rope with built-in digital rep counter and timer.", 14.99, _local("sports/DigitalJumpRopeCounter.png")),
        ("Wooden Gymnastic Rings",         "Birch gymnastic rings with 4.5m adjustable nylon straps.", 29.99, _local("sports/WoodenGymnasitcRings.png")),
        ("Ankle Weights Pair 2kg",         "Adjustable neoprene ankle weights, 1kg per ankle.", 17.99, _local("sports/AnkleWeights.png")),
        ("Ab Wheel Roller",                "Double-wheel ab roller with non-slip handles and kneeling pad.", 19.99, _local("sports/AbWheelRoller.png")),
        ("Climbing Chalk Block 57g",       "MgCO₃ gym chalk block — maximise grip, minimise sweat.", 7.99,  _local("sports/ClimbingChalk.png")),
        ("Tennis Racket",                  "Alloy-frame intermediate racket, 102 sq inch head, pre-strung.", 59.99, _local("sports/TennisRacket.png")),
        ("Badminton Set",                  "4 rackets + 3 shuttles + net, all-in-one carry bag.", 29.99, _local("sports/BadmintonSet.png")),
        ("Football Match Ball Size 5",     "FIFA Inspected hand-stitched ball, 32-panel thermally bonded.", 34.99, _local("sports/FootballMatchBall.png")),
        ("Basketball Size 7",             "Official-size rubber basketball, indoor/outdoor use.", 44.99, _local("sports/Basketball.png")),
        ("Road Cycling Helmet",            "Aerodynamic MIPS helmet, 18 vents, fits 52-58cm.", 69.99, _local("sports/CyclingHelmet.png")),
        ("Heavy-Duty Bike Lock",           "Folding hardened-steel lock with mounting bracket, ART ★★.", 27.99, _local("sports/BikeLock.png")),
        ("Waterproof Running Jacket",      "Seamless 3-layer waterproof jacket, packable, reflective trim.", 79.99, _local("sports/WaterproofRunningJacketz.png")),
        ("Compression Knee Sleeve",        "Neoprene sleeve with open patella, graduated compression.", 19.99, _local("sports/CompressionKneeSleve.png")),
        ("GPS Fitness Tracker Band",       "24/7 heart-rate, step, sleep, and SpO₂ tracking, 7-day battery.", 49.99, _local("sports/GpsFitnessTracker.png")),
        ("Acupressure Mat and Pillow Set", "Lotus spike mat (66x40cm) with neck pillow, cotton backing.", 34.99, _local("sports/AcupressureMatandPillow.png")),
        ("Balance Board Wobble",           "360° balance board, anti-slip surface, 150kg rated.", 39.99, _local("sports/BalanceWoodBoard.png")),
        ("Collapsible Hiking Poles Pair",  "Aluminium 3-section twist-lock poles with cork grips.", 54.99, _local("sports/HikingPolesPair.png")),
        ("Camping Hammock",                "Lightweight ripstop-nylon hammock, 280kg tested, 300g.", 44.99, _local("sports/Hammock.png")),
        ("Hydration Running Vest 10L",     "Soft-flask vest with 2L bladder sleeve, 10L storage.", 69.99, _local("sports/HydrationRunningVest.png")),
        ("Running Armband Phone Holder",   "Stretch lycra armband with touchscreen-compatible window.", 14.99,_local("sports/RunningArmbandPhoneHolder.png")),
        ("Resistance Loop Bands 10-Pack",  "Fabric-reinforced loop bands in 10 progressive resistances.", 19.99, _local("sports/ResistanceBands10pack.png")),
        ("Lacrosse Ball Massage 3-Pack",   "High-density rubber massage balls for deep-tissue release.", 12.99, _local("sports/LacrossBalls.png")),
        ("Speed Agility Ladder",           "12-rung 6m agility ladder with 4 metal stakes and carry bag.", 19.99, _local("sports/SpeedAgilityLadder.png")),
        ("Stopwatch Digital",              "Lap-memory stopwatch with split-time and countdown functions.", 14.99, _local("sports/StopWatch.png")),
        ("Sports Water Bottle 2L",         "Tritan BPA-free wide-mouth sports bottle with leak-proof lid.", 24.99, _local("sports/2LWaterBottle.png")),
        ("Swimming Goggles",               "UV400 anti-fog goggles with adjustable silicone strap.", 19.99, _local("sports/SwimGoggles.png")),
        ("Silicone Swim Cap",              "Competition silicone cap, unisex, tear-resistant.", 9.99,  _local("sports/SwimCap.png")),
        ("Grip Strengthener",              "Adjustable resistance hand gripper, 10-40kg range.", 12.99, _local("sports/GripStrengthener.png")),
        ("Heavy Punching Bag 25kg",        "Filled heavy bag with hanging chains and D-ring, 25kg.", 89.99, _local("sports/PunchingBag.png")),
        ("Boxing Gloves 10oz",             "Hook-and-loop training gloves, foam padding, synthetic leather.", 44.99, _local("sports/BoxingGloves.png")),
        ("Roller Skates Adults",           "Quad skates with adjustable buckle, 82A wheels.", 69.99, _local("sports/RollerSkates.png")),
        ("Complete Skateboard",            "7-ply maple deck, ABEC-7 bearings, 52mm wheels.", 59.99, _local("sports/Skateboard.png")),
        ("Yoga Blocks Set of 2",           "High-density EVA foam blocks with bevelled edges.", 19.99, _local("sports/YogaBlocks.png")),
        ("Stretching Strap 10-Loop",       "Cotton stretching strap with 10 loops for assisted stretching.", 9.99,  _local("sports/StretchingStrap.png")),
        ("Pilates Ring",                   "38cm flexible foam-padded pilates ring, 6.5kg resistance.", 19.99, _local("sports/PilatesRing.png")),
        ("Core Exercise Sliders 2-Pack",   "Dual-sided fitness sliders — carpet and hardwood compatible.", 12.99, _local("sports/CoreExerciseSlider.png")),
        ("TRX Suspension Trainer",         "Full-body suspension trainer with anchoring kit, 350lb rated.", 79.99, _local("sports/SuspensionTrainer.png")),
        ("Dipping Belt Leather",           "Genuine leather weight dipping belt with 100cm chain.", 34.99, _local("sports/DippingBeltLeather.png")),
        ("Olympic Weight Plate Pair 5kg",  "Rubber-coated 50mm hole Olympic plates, colour-coded.", 29.99, _local("sports/OlympicWeightPlatePair5kg.png")),
    ],

    # ── Accessories ───────────────────────────────────────────────────────────
    "Accessories": [
        ("Minimalist Watch",              "Stainless-steel case, Japanese quartz movement, leather strap.", 89.99, _local("accessories/MinimalistWatch.png")),
        ("Slim RFID Wallet",              "Full-grain leather bifold with RFID-blocking carbon-fibre insert.", 34.99, _local("accessories/RFIDWallter.png")),
        ("Canvas Backpack 30L",           "Waxed-canvas rucksack with 15\" laptop sleeve and organiser.", 59.99, _local("accessories/CanvasBackpag.png")),
        ("Polarized Sunglasses",          "UV400 polarized lenses, acetate frame, spring hinges.", 44.99, _local("accessories/Sunnglases.png")),
        ("Shockproof Phone Case",         "Dual-layer TPU and polycarbonate case, MagSafe compatible.", 19.99, _local("accessories/ShockproofPhoneCase.png")),
        ("Keychain Multi-tool 10-in-1",   "Compact stainless steel EDC multi-tool that clips to any keyring.", 14.99, _local("accessories/KeychainMultiTool.png")),
        ("Full-Grain Leather Belt",       "1.5\" full-grain belt with antique-brass single-prong buckle.", 29.99, _local("accessories/LeatherBelt.png")),
        ("Leather Passport Holder",       "Vegetable-tanned passport cover with card and document slots.", 17.99, _local("accessories/LeatherPassportHolder.png")),
        ("Laptop Bag 15 inch",            "Padded crossbody bag, water-repellent, fits up to 15.6\".", 49.99, _local("accessories/LaptopBag.png")),
        ("Titanium Card Holder",          "Machined titanium pull-tab cardholder, holds 8 cards.", 29.99, _local("accessories/BifoldCardholder.png")),
        ("Silk Pocket Square",            "Handrolled pure-silk pocket square in classic solid colours.", 19.99, _local("accessories/PocketSquare.png")),
        ("Leather Driving Gloves",        "Perforated leather gloves with snap fastener, unlined.", 34.99, _local("accessories/LeatherDrivingGloves.png")),
        ("Roll-Up Straw Hat",             "UPF 50+ paper-straw hat, crushable, with grosgrain ribbon.", 22.99, _local("accessories/Roll-UpStrawHat.png")),
        ("Gold-Plated Cufflinks",         "Knot-style cufflinks in 18k gold-plated brass with gift box.", 24.99, _local("accessories/Gold-PlatedCufflinks.png")),
        ("AirPods Leather Case",          "Vegetable-tanned leather case for AirPods Gen 1-4 with keyring.", 19.99, _local("accessories/AirPodsLeatherCase.png")),
        ("Compact Auto Umbrella",         "Windproof 8-rib auto open/close umbrella, 23cm folded.", 24.99, _local("accessories/compactAutoUbmrella.png")),
        ("Canvas Tote Bag",               "12oz natural canvas tote with long handles and interior pocket.", 22.99, _local("accessories/CanvasToteBag.png")),
        ("Duffel Bag 40L",                "Waxed canvas duffel with shoe compartment and shoulder strap.", 69.99, _local("accessories/DuffelBag40L.png")),
        ("RFID Blocking Card",            "Passive RFID blocker card — protects whole wallet, no battery.", 9.99,  _local("accessories/RFIDBlockingCard.png")),
        ("Universal Travel Adapter",      "Compact all-in-one adapter with 3 USB-A + 1 USB-C ports.", 27.99, _local("accessories/TravelAdapter.png")),
        ("Memory Foam Neck Pillow",       "Ergonomic U-shaped travel pillow with machine-washable cover.", 29.99, _local("accessories/MemoryFoamNeckPillow.png")),
        ("Luggage Tag Set 3-Pack",        "Genuine leather tags with privacy flap, 3 colours.", 12.99, _local("accessories/LuggageTagSet.png")),
        ("Leather Watch Band 22mm",       "Crocodile-grain genuine leather NATO strap, quick-release.", 19.99, _local("accessories/LeatherWatchBand.png")),
        ("Stainless Money Clip",          "Slim stainless steel money clip, brushed finish.", 14.99, _local("accessories/StainlessMoneyClip.png")),
        ("Leather Crossbody Bag",         "Pebbled leather crossbody with adjustable strap, 4 pockets.", 49.99, _local("accessories/LeatherCrossbodyBag.png")),
        ("Beaded Bracelet Set",           "Set of 6 natural-stone bead bracelets on elastic cord.", 17.99, _local("accessories/BeadedBraceletSet.png")),
        ("Carabiner Clips 5-Pack",        "12kN aluminium D-ring carabiners — keyring, bag, outdoor.", 12.99, _local("accessories/CarabinerClips5Pack.png")),
        ("Retractable Badge Reel Lanyard","Swivel spring reel with 90cm extend, belt clip.", 7.99,  _local("accessories/RetractableBadgeReelLanyard.png")),
        ("Clip-On Sunglasses",            "Magnetic clip-on UV400 lenses for prescription frames.", 19.99, _local("accessories/Clip-OnSunglasses.png")),
        ("Waterproof Toiletry Bag",       "Hanging roll bag with 8 pockets, YKK zips, ripstop nylon.", 22.99, _local("accessories/WaterproofToiletryBag.png")),
        ("Wristlet Clutch",               "Pebbled vegan-leather clutch with wrist strap and card slots.", 29.99, _local("accessories/WristletClutch.png")),
        ("Genuine Leather Coin Purse",    "Compact zip purse in nappa leather, 10x8cm.", 17.99, _local("accessories/LeathercoinPruse.png")),
        ("Metal Business Card Holder",    "Brushed stainless card holder, holds 20 cards.", 14.99, _local("accessories/MetalBusinessCardHolder.png")),
        ("Retro Fanny Pack",              "90s-style ripstop fanny pack with buckle strap.", 24.99, _local("accessories/RetroFannyBag.png")),
        ("Bifold Cardholder",             "Slim vegan-leather cardholder with 8 slots and cash sleeve.", 19.99, _local("accessories/BifoldCardholder.png")),
        ("Slim Key Organiser",            "Aircraft-aluminium key holder, stacks up to 12 keys flat.", 22.99, _local("accessories/SlimKeyOrganizer.png")),
        ("Wireless Watch Charging Stand", "MagSafe-style watch charging dock for Apple Watch S1-10.", 17.99, _local("accessories/WirelessWatchChargingStand.png")),
        ("Reading Glasses Plus 1.5",      "Spring-hinge reading glasses +1.5 dioptre in anti-glare.", 14.99, _local("accessories/ReadingGlassesPlus.png")),
        ("Clip-On Polarized Glasses",     "Flip-up polarized clip-on for wire-rim frames, universal fit.", 22.99, _local("accessories/Clip-OnPolarizedGlasses.png")),
        ("Handmade Braided Bracelet",     "Hand-knotted macramé bracelet in natural cotton cord.", 19.99, _local("accessories/HandmadeBraidedBracelet.png")),
        ("Silk Tie",                      "100% silk jacquard tie, 8cm blade, hand-finished tip.", 29.99, _local("accessories/SilkTie.png")),
        ("Knitted Bow Tie",               "Pre-tied knitted bow tie in Shetland wool, adjustable band.", 19.99, _local("accessories/KnittedBowTie.png")),
        ("Snapback Baseball Cap",         "6-panel structured cap with flat brim and embroidered logo.", 22.99, _local("accessories/SnapbackBaseballCap.png")),
        ("Wide Brim Felt Hat",            "100% wool felt fedora with grosgrain band, crushable crown.", 44.99, _local("accessories/WideBrimFeltHat.png")),
        ("Leather Watch Roll 3-Slot",     "Vegetable-tanned leather watch roll with suede lining.", 39.99, _local("accessories/LeatherWatchRoll.png")),
        ("Travel Jewelry Roll",           "Padded organiser roll with mirror, 6 pockets, 2 ring bars.", 24.99, _local("accessories/TravelJewelryRoll.png")),
        ("Slim Backpack 15L",             "Streamlined 15L daypack, water-repellent, padded back panel.", 44.99, _local("accessories/SlimBackpack15L.png")),
        ("Anti-Theft Shoulder Bag",       "RFID-blocking shoulder bag with slash-resistant straps.", 54.99, _local("accessories/Anti-TheftSchoulderBag.png")),
        ("Hard Sunglasses Case",          "Clamshell hard case with microfibre lining and belt loop.", 12.99, _local("accessories/HardSunglassesCase.png")),
        ("Cable Organiser Wallet",        "Zippered cable roll with 8 elastic loops for cables and adapters.", 19.99, _local("accessories/CableOrganizerWallet.png")),
    ],
}


def reconcile_product_images() -> int:
    """Re-point existing products at their current seed image path.

    Seeding is skipped entirely once the database has rows, so a deployed
    instance keeps whatever ``image_url`` it was first seeded with. If an
    image is subsequently renamed in the repository, that stale path stops
    resolving and the product silently renders a broken image -- with the
    derivative pipeline it also means no WebP variant exists under the
    stored name.

    Matching on ``slug`` (stable, derived from the product name) this
    realigns the stored paths with ``PRODUCTS`` on every startup. Only
    ``image_url`` is touched, so collected orders, events and decision logs
    are unaffected.

    Returns:
        The number of products whose image path was corrected.
    """
    expected = {
        slugify(name): image_url
        for products in PRODUCTS.values()
        for name, _description, _price, image_url in products
    }

    db = SessionLocal()
    try:
        fixed = 0
        for product in db.query(Product).all():
            want = expected.get(product.slug)
            if want and product.image_url != want:
                product.image_url = want
                fixed += 1
        if fixed:
            db.commit()
        return fixed
    finally:
        db.close()


def seed_database(force: bool = False) -> bool:
    """Populate the database with categories and products.

    Args:
        force: If True, clear existing products/categories before inserting.

    Returns:
        True if seeding was performed, False if skipped.
    """
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if not force and db.query(Category).count() > 0:
            return False  # Already seeded — skip silently

        if force:
            # Delete in FK-safe order (cart/orders reference products)
            db.query(CartItem).delete()
            db.query(OrderItem).delete()
            db.query(Order).delete()
            db.query(Product).delete()
            db.query(Category).delete()
            db.commit()

        categories: dict[str, int] = {}
        for cat_data in CATEGORIES:
            cat = Category(
                name=cat_data["name"],
                slug=slugify(cat_data["name"]),
                description=cat_data["description"],
            )
            db.add(cat)
            db.flush()
            categories[cat_data["name"]] = cat.id

        for cat_name, products in PRODUCTS.items():
            for name, description, price, image_url in products:
                product = Product(
                    name=name,
                    slug=slugify(name),
                    description=description,
                    price=price,
                    image_url=image_url,
                    category_id=categories[cat_name],
                    stock=100,
                )
                db.add(product)

        db.commit()
        return True
    finally:
        db.close()


if __name__ == "__main__":
    force = "--force" in sys.argv
    result = seed_database(force=force)
    if result:
        action = "re-seeded" if force else "seeded"
        print(f"Database {action} successfully — {sum(len(v) for v in PRODUCTS.values())} products across {len(CATEGORIES)} categories.")
    else:
        print("Database already populated. Run with --force to reseed.")
