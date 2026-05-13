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
]
