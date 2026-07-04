# TASK2 sibling clusters (annotation-free, text cosine CC)

- source: /workspace/Mosaic3D/error_analysis/reports/text_embeddings.npz
- foreground classes: 197

## threshold = 0.85
- clusters=2, multi-member clusters=1, classes covered by multi-member clusters=196

| cluster | size | members |
|---|---:|---|
| 1 | 196 | chair, table, door, couch, cabinet, shelf, desk, office chair, bed, pillow, sink, picture, window, toilet, bookshelf, monitor, curtain, book, armchair, coffee table, box, refrigerator, lamp, kitchen cabinet, towel, clothes, tv, nightstand, counter, dresser, stool, cushion, plant, bathtub, end table, dining table, keyboard, bag, backpack, toilet paper, printer, tv stand, whiteboard, blanket, shower curtain, trash can, closet, stairs, microwave, stove, shoe, computer tower, bottle, bin, ottoman, bench, board, washing machine, mirror, copier, basket, sofa chair, file cabinet, fan, laptop, shower, paper, person, paper towel dispenser, oven, blinds, rack, plate, blackboard, piano, suitcase, rail, radiator, recycling bin, container, wardrobe, soap dispenser, telephone, bucket, clock, stand, light, laundry basket, pipe, clothes dryer, guitar, toilet paper holder, seat, speaker, column, bicycle, ladder, bathroom stall, shower wall, cup, jacket, storage bin, coffee maker, dishwasher, paper towel roll, machine, mat, windowsill, bar, toaster, bulletin board, ironing board, fireplace, soap dish, kitchen counter, doorframe, toilet paper dispenser, mini fridge, fire extinguisher, ball, hat, shower curtain rod, water cooler, paper cutter, tray, shower door, pillar, ledge, toaster oven, mouse, toilet seat cover dispenser, furniture, cart, storage container, scale, tissue box, light switch, crate, power outlet, decoration, sign, projector, closet door, vacuum cleaner, candle, plunger, stuffed animal, headphones, dish rack, broom, guitar case, range hood, dustpan, hair dryer, water bottle, handicap bar, purse, vent, shower floor, water pitcher, mailbox, bowl, paper bag, alarm clock, music stand, projector screen, divider, laundry detergent, bathroom counter, object, bathroom vanity, closet wall, laundry hamper, bathroom stall door, ceiling light, trash bin, stair rail, tube, bathroom cabinet, cd case, closet rod, coffee kettle, structure, shower head, keyboard piano, case of water bottles, coat rack, storage organizer, folded chair, fire alarm, power strip, calendar, poster, potted plant, luggage, mattress |

## threshold = 0.88
- clusters=29, multi-member clusters=5, classes covered by multi-member clusters=173

| cluster | size | members |
|---|---:|---|
| 1 | 163 | chair, table, door, couch, cabinet, shelf, desk, office chair, bed, pillow, sink, picture, window, toilet, bookshelf, monitor, curtain, book, armchair, coffee table, box, refrigerator, lamp, kitchen cabinet, towel, clothes, tv, nightstand, counter, dresser, stool, cushion, bathtub, end table, dining table, keyboard, bag, backpack, toilet paper, printer, tv stand, whiteboard, blanket, shower curtain, trash can, closet, microwave, stove, computer tower, bottle, bin, bench, board, washing machine, mirror, copier, basket, sofa chair, file cabinet, laptop, shower, paper, person, paper towel dispenser, oven, blinds, rack, plate, blackboard, piano, suitcase, recycling bin, container, wardrobe, soap dispenser, telephone, bucket, clock, stand, light, laundry basket, clothes dryer, guitar, toilet paper holder, seat, speaker, column, bathroom stall, shower wall, cup, jacket, storage bin, coffee maker, dishwasher, paper towel roll, machine, mat, windowsill, bar, toaster, bulletin board, fireplace, kitchen counter, doorframe, toilet paper dispenser, mini fridge, ball, shower curtain rod, water cooler, tray, shower door, pillar, toaster oven, toilet seat cover dispenser, furniture, storage container, scale, tissue box, light switch, crate, power outlet, decoration, sign, projector, closet door, candle, headphones, dish rack, guitar case, range hood, water bottle, handicap bar, purse, shower floor, water pitcher, mailbox, bowl, paper bag, alarm clock, music stand, projector screen, laundry detergent, bathroom counter, object, bathroom vanity, closet wall, laundry hamper, bathroom stall door, ceiling light, trash bin, bathroom cabinet, closet rod, coffee kettle, structure, shower head, keyboard piano, case of water bottles, storage organizer, folded chair, power strip, poster, luggage, mattress |
| 50 | 4 | stairs, rail, ladder, stair rail |
| 34 | 2 | plant, potted plant |
| 91 | 2 | pipe, tube |
| 121 | 2 | fire extinguisher, fire alarm |

## threshold = 0.90
- clusters=70, multi-member clusters=18, classes covered by multi-member clusters=145

| cluster | size | members |
|---|---:|---|
| 1 | 101 | chair, table, door, couch, cabinet, desk, office chair, bed, sink, picture, window, toilet, monitor, book, armchair, coffee table, box, lamp, kitchen cabinet, clothes, tv, nightstand, counter, dresser, stool, bathtub, end table, dining table, keyboard, bag, backpack, toilet paper, tv stand, trash can, closet, microwave, stove, bin, bench, washing machine, basket, sofa chair, file cabinet, laptop, paper, person, paper towel dispenser, oven, piano, suitcase, recycling bin, container, wardrobe, soap dispenser, bucket, stand, light, laundry basket, clothes dryer, guitar, toilet paper holder, seat, jacket, storage bin, dishwasher, paper towel roll, machine, windowsill, bar, toaster, fireplace, kitchen counter, doorframe, toilet paper dispenser, toaster oven, furniture, storage container, crate, decoration, projector, closet door, candle, guitar case, purse, paper bag, projector screen, bathroom counter, object, bathroom vanity, closet wall, laundry hamper, ceiling light, trash bin, bathroom cabinet, closet rod, structure, keyboard piano, storage organizer, folded chair, luggage, mattress |
| 68 | 5 | shower, shower wall, shower door, shower floor, shower head |
| 75 | 4 | plate, cup, tray, bowl |
| 18 | 3 | curtain, shower curtain, shower curtain rod |
| 55 | 3 | bottle, water bottle, case of water bottles |
| 59 | 3 | board, sign, poster |
| 105 | 3 | coffee maker, water pitcher, coffee kettle |
| 139 | 3 | light switch, power outlet, power strip |
| 7 | 2 | shelf, bookshelf |
| 11 | 2 | pillow, cushion |
| 23 | 2 | refrigerator, mini fridge |
| 26 | 2 | towel, blanket |
| 34 | 2 | plant, potted plant |
| 43 | 2 | printer, copier |
| 50 | 2 | stairs, stair rail |
| 87 | 2 | clock, alarm clock |
| 97 | 2 | column, pillar |
| 100 | 2 | bathroom stall, bathroom stall door |
