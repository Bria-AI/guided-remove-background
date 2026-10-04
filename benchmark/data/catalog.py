"""Curated image catalog for guided RMBG benchmark.

Two scenario types:
  A. Ambiguous foreground — no clear single subject, RMBG doesn't know what to cut
  B. Adjustable foreground — RMBG picks a default, user wants to adjust scope
"""

CATALOG = [
    # === Scenario A: Ambiguous foreground ===

    ("living_room.jpg",
     "https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Open-plan living room: L-shaped sofa, gold coffee table, patterned rug, chandelier, staircase, kitchen in background"),

    ("smoothie_bowl.jpg",
     "https://images.pexels.com/photos/1099680/pexels-photo-1099680.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Top-down smoothie bowl: berry smoothie bowl with raspberries, blackberries, almonds, mango, coconut; scattered fruits around — mango halves, strawberries, blueberries, herbs, spoon"),

    ("art_table.jpg",
     "https://images.pexels.com/photos/1053687/pexels-photo-1053687.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Top-down art workspace: wooden paint palette with mixed colors, brushes in glass jar, blank canvas with pencil, paint tubes, pastel set, stained rag on wooden table"),

    ("plant_shelf.jpg",
     "https://images.pexels.com/photos/4503273/pexels-photo-4503273.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Row of labeled herb pots on wooden shelf: pepper, mint, tomato, basil, oregano in terracotta pots of varying sizes, clean white wall background"),

    ("kids_room.jpg",
     "https://images.pexels.com/photos/1648768/pexels-photo-1648768.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Kid's room: white daybed with blue pillows and stuffed animals, tree wall art, desk with iMac, bookshelf, chandelier, rug"),

    ("desk_setup.jpg",
     "https://images.pexels.com/photos/1006293/pexels-photo-1006293.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Desk scene: Dell laptop open on desk, potted plant in red pot, reading glasses on notebook, smartphone, white curtain background"),

    ("garden_patio.jpg",
     "https://images.pexels.com/photos/8916602/pexels-photo-8916602.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Mediterranean garden patio: table with white cloth, flower vase with wildflowers, fruit bowl, colored glasses, open book, chairs with floral cushions, hanging Moroccan lantern, bougainvillea pergola above"),

    ("cafe_interior.jpg",
     "https://images.pexels.com/photos/1307698/pexels-photo-1307698.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Surf café interior: wooden tables and chairs, bar counter with bottles, blue surfboard, potted plants, framed wave photos, pendant lights"),

    ("cafe_table.jpg",
     "https://images.pexels.com/photos/2074130/pexels-photo-2074130.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Top-down café table: multiple coffee cups on blue saucers, passion fruit, cookies, book, phone, glass teapots, hands reaching in from edges"),

    # === Scenario B: Adjustable foreground ===

    ("office_meeting.jpg",
     "https://images.pexels.com/photos/3184291/pexels-photo-3184291.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Six people around meeting table: standing woman shaking hands, laptops, coffee cups, sticky notes, cork board behind"),

    ("person_dog.jpg",
     "https://images.pexels.com/photos/1612847/pexels-photo-1612847.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Woman in blue jacket walking dog on autumn forest path, fallen leaves, ferns, trees"),

    ("cooking_scene.jpg",
     "https://images.pexels.com/photos/28703300/pexels-photo-28703300.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Chef in apron cooking at kitchen counter: multiple pots and pans on stove, plates, ingredients, tiled backsplash, bottles"),

    ("home_office.jpg",
     "https://images.pexels.com/photos/4050315/pexels-photo-4050315.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Top-down view: woman at white desk with laptop, phone, open book, coffee mug, glasses, woven chair, rug on wooden floor"),

    ("yoga_studio.jpg",
     "https://images.pexels.com/photos/3822906/pexels-photo-3822906.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Woman in yoga pose on galaxy-print mat, white candles, tall palm plant, marble wall, large window, wooden floor"),

    ("patio_bar.jpg",
     "https://images.pexels.com/photos/1267696/pexels-photo-1267696.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Two women laughing at outdoor brewery patio with beer glasses, wooden bench, third person partially visible, bar shelves and street behind"),

    # === AG-194 additions ===
    # Picked to be simple and clearly visible: one clear subject or a few well-separated
    # objects, a good camera angle, an uncluttered background. Varied on purpose across
    # situations (studio, home, outdoor nature, street), angles (eye-level, three-quarter,
    # top-down) and backgrounds, so alpha edges and identity preservation are easy to judge.

    ("sneakers_on_boxes.jpg",
     "https://images.pexels.com/photos/4273288/pexels-photo-4273288.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Pair of blue high-top sneakers with orange laces on top of two stacked black Nike shoe boxes, blue wall"),

    ("sneakers_plant.jpg",
     "https://images.pexels.com/photos/16604058/pexels-photo-16604058.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Pair of tan suede sneakers beside a small potted topiary plant on a white studio background"),

    ("dropper_bottles_rocks.jpg",
     "https://images.pexels.com/photos/6767761/pexels-photo-6767761.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Three skincare dropper bottles standing on flat rocks against a terracotta and white backdrop"),

    ("watch_flatlay.jpg",
     "https://images.pexels.com/photos/1906607/pexels-photo-1906607.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Top-down flat lay: steel wristwatch, glasses, cup of coffee and a potted succulent on a round wooden board"),

    ("man_pug_mug.jpg",
     "https://images.pexels.com/photos/7788915/pexels-photo-7788915.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Smiling man in a white T-shirt holding a black pug in one arm and a white mug in the other, white tiled kitchen"),

    ("man_bench_mountains.jpg",
     "https://images.pexels.com/photos/842291/pexels-photo-842291.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Bearded man in sunglasses sitting on a wooden bench, green mountain valley and sky behind"),

    ("courier_bench.jpg",
     "https://images.pexels.com/photos/8988546/pexels-photo-8988546.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "adjustable", "Courier sitting on a blue street bench with a teal thermal delivery bag beside him and a white bicycle behind, glass building and shrubs"),

    ("bedroom_desk.jpg",
     "https://images.pexels.com/photos/6782479/pexels-photo-6782479.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Modern bedroom: bed with mint headboard and grey pillows, upholstered chair at a built-in desk, wall shelving above"),

    ("armchair_lamp.jpg",
     "https://images.pexels.com/photos/6078545/pexels-photo-6078545.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Red armchair with a headrest next to a tall white floor lamp, plain white wall, tiled floor"),

    ("reading_corner.jpg",
     "https://images.pexels.com/photos/18338530/pexels-photo-18338530.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Reading corner: blue armchair, small side table, yellow floor lamp, wooden shelf with a potted palm, panelled wall, rug"),

    ("breakfast_plate.jpg",
     "https://images.pexels.com/photos/36842543/pexels-photo-36842543.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Top-down minimal breakfast on a white table: plate with boiled eggs, bread and butter, egg in a cup, cup of coffee, cutlery"),

    ("laptop_pears.jpg",
     "https://images.pexels.com/photos/7214935/pexels-photo-7214935.jpeg?auto=compress&cs=tinysrgb&w=1280",
     "ambiguous", "Open laptop with a blank white screen next to two green pears on a white table, plain white wall"),
]

# Catalog images no longer in the active benchmark (cases.csv): too busy, or replaced by a
# cleaner image of the same kind. Kept so the Sep 2026 snapshot (benchmark/results/sep-2026/)
# stays reproducible.
RETIRED = {
    "living_room.jpg", "smoothie_bowl.jpg", "art_table.jpg", "plant_shelf.jpg", "kids_room.jpg",
    "garden_patio.jpg", "cafe_interior.jpg", "cafe_table.jpg", "office_meeting.jpg",
    "person_dog.jpg", "cooking_scene.jpg", "patio_bar.jpg",
}

# Every image is from Pexels and used under the Pexels License (free to use, attribution not
# required, may not be sold unaltered or redistributed as a stock collection).
LICENSE = "Pexels License (https://www.pexels.com/license/)"

# Source page and photographer as listed on Pexels. The original 15 were added without a
# recorded photographer; their photo pages still resolve by id.
SOURCES = {
    "living_room.jpg": ("https://www.pexels.com/photo/1571460/", None),
    "smoothie_bowl.jpg": ("https://www.pexels.com/photo/1099680/", None),
    "art_table.jpg": ("https://www.pexels.com/photo/1053687/", None),
    "plant_shelf.jpg": ("https://www.pexels.com/photo/4503273/", None),
    "kids_room.jpg": ("https://www.pexels.com/photo/1648768/", None),
    "desk_setup.jpg": ("https://www.pexels.com/photo/1006293/", None),
    "garden_patio.jpg": ("https://www.pexels.com/photo/8916602/", None),
    "cafe_interior.jpg": ("https://www.pexels.com/photo/1307698/", None),
    "cafe_table.jpg": ("https://www.pexels.com/photo/2074130/", None),
    "office_meeting.jpg": ("https://www.pexels.com/photo/3184291/", None),
    "person_dog.jpg": ("https://www.pexels.com/photo/1612847/", None),
    "cooking_scene.jpg": ("https://www.pexels.com/photo/28703300/", None),
    "home_office.jpg": ("https://www.pexels.com/photo/4050315/", None),
    "yoga_studio.jpg": ("https://www.pexels.com/photo/3822906/", None),
    "patio_bar.jpg": ("https://www.pexels.com/photo/1267696/", None),
    "sneakers_on_boxes.jpg": ("https://www.pexels.com/photo/photo-of-high-top-sneakers-on-nike-box-4273288/", "Introspectivedsgn"),
    "sneakers_plant.jpg": ("https://www.pexels.com/photo/shoes-and-plant-on-white-background-16604058/", "Rubaitulazad"),
    "dropper_bottles_rocks.jpg": ("https://www.pexels.com/photo/a-skin-care-products-on-a-glass-bottles-with-droppers-6767761/", "misolo-cosmetic"),
    "watch_flatlay.jpg": ("https://www.pexels.com/photo/a-cup-of-coffee-stainless-watch-and-eyeglasses-on-a-table-with-ornamental-plant-1906607/", "Nietjuhart"),
    "man_pug_mug.jpg": ("https://www.pexels.com/photo/young-man-holding-a-pug-and-a-cup-of-coffee-7788915/", "Babydov"),
    "man_bench_mountains.jpg": ("https://www.pexels.com/photo/photography-of-a-man-sitting-on-wooden-bench-842291/", "Annetnavi"),
    "courier_bench.jpg": ("https://www.pexels.com/photo/man-in-white-dress-shirt-sitting-on-blue-bench-beside-a-thermal-bag-8988546/", "Artem Podrez"),
    "bedroom_desk.jpg": ("https://www.pexels.com/photo/apartment-interior-with-bed-near-table-with-chair-6782479/", "Artbovich"),
    "armchair_lamp.jpg": ("https://www.pexels.com/photo/comfortable-armchair-and-lamp-near-white-wall-6078545/", "Artbovich"),
    "reading_corner.jpg": ("https://www.pexels.com/photo/lamp-plant-and-armchair-in-room-18338530/", "Ekru Lila"),
    "breakfast_plate.jpg": ("https://www.pexels.com/photo/minimalist-breakfast-with-coffee-and-eggs-36842543/", "ekrulila"),
    "laptop_pears.jpg": ("https://www.pexels.com/photo/a-laptop-with-a-white-screen-7214935/", "Anna Nekrashevich"),
}

# Benchmark class per active image, the grouping AG-195 reports win rates by.
IMAGE_CLASS = {
    "yoga_studio.jpg": "people", "home_office.jpg": "people", "man_pug_mug.jpg": "people",
    "man_bench_mountains.jpg": "people", "courier_bench.jpg": "people",
    "sneakers_on_boxes.jpg": "product", "sneakers_plant.jpg": "product",
    "dropper_bottles_rocks.jpg": "product", "watch_flatlay.jpg": "product",
    "bedroom_desk.jpg": "interior", "armchair_lamp.jpg": "interior", "reading_corner.jpg": "interior",
    "desk_setup.jpg": "multi_object", "laptop_pears.jpg": "multi_object", "breakfast_plate.jpg": "multi_object",
}
