# FN destinations (top 25 target classes by cost)

For each class C, where did its GT points get predicted (excluding the correct TP)? High top-1 share => synonym/confusion; diffuse => feature/recall problem.

## chair (idx=1)
- gt=2491617 pred=1314846 IoU=0.4915 recall=0.5034 precision=0.9540 cost=1237265
- FN total=1237265, top-1 destination share=54.6%
    - office chair: 676121 (54.6%)
    - folded chair: 136409 (11.0%)
    - table: 85940 (6.9%)
    - stool: 74589 (6.0%)
    - seat: 69365 (5.6%)
    - dining table: 49230 (4.0%)
    - armchair: 36148 (2.9%)
    - sofa chair: 12843 (1.0%)

## shelf (idx=7)
- gt=1169879 pred=626609 IoU=0.2499 recall=0.3070 precision=0.5733 cost=810674
- FN total=810675, top-1 destination share=45.5%
    - bookshelf: 368567 (45.5%)
    - storage organizer: 109297 (13.5%)
    - rack: 60749 (7.5%)
    - closet: 59021 (7.3%)
    - closet wall: 28561 (3.5%)
    - case of water bottles: 21632 (2.7%)
    - desk: 19165 (2.4%)
    - computer tower: 16442 (2.0%)

## cabinet (idx=6)
- gt=758107 pred=207686 IoU=0.1603 recall=0.1760 precision=0.6424 cost=624692
- FN total=624692, top-1 destination share=14.5%
    - kitchen cabinet: 90547 (14.5%)
    - dresser: 45464 (7.3%)
    - closet: 45279 (7.2%)
    - shelf: 41507 (6.6%)
    - closet wall: 35763 (5.7%)
    - file cabinet: 33102 (5.3%)
    - counter: 29909 (4.8%)
    - door: 26734 (4.3%)

## window (idx=14)
- gt=1385131 pred=1096729 IoU=0.4969 recall=0.5948 precision=0.7512 cost=561321
- FN total=561322, top-1 destination share=36.9%
    - blinds: 207153 (36.9%)
    - door: 82788 (14.7%)
    - windowsill: 32161 (5.7%)
    - curtain: 25610 (4.6%)
    - whiteboard: 20704 (3.7%)
    - board: 16777 (3.0%)
    - mirror: 14777 (2.6%)
    - doorframe: 11519 (2.1%)

## door (idx=4)
- gt=1291634 pred=1139657 IoU=0.5281 recall=0.6505 precision=0.7373 cost=451372
- FN total=451373, top-1 destination share=21.9%
    - window: 98980 (21.9%)
    - closet door: 94308 (20.9%)
    - bathroom stall door: 46614 (10.3%)
    - closet: 43220 (9.6%)
    - closet wall: 42524 (9.4%)
    - shower door: 29975 (6.6%)
    - doorframe: 10464 (2.3%)
    - bathroom stall: 8073 (1.8%)

## table (idx=3)
- gt=1224698 pred=1070620 IoU=0.5898 recall=0.6953 precision=0.7954 cost=373151
- FN total=373151, top-1 destination share=28.3%
    - desk: 105609 (28.3%)
    - stool: 32999 (8.8%)
    - dining table: 25199 (6.8%)
    - coffee table: 24051 (6.4%)
    - counter: 20123 (5.4%)
    - office chair: 19289 (5.2%)
    - bench: 16737 (4.5%)
    - chair: 8808 (2.4%)

## book (idx=19)
- gt=352186 pred=77 IoU=0.0001 recall=0.0001 precision=0.6104 cost=352139
- FN total=352139, top-1 destination share=91.6%
    - bookshelf: 322560 (91.6%)
    - paper bag: 3992 (1.1%)
    - shelf: 2402 (0.7%)
    - table: 2326 (0.7%)
    - laptop: 2028 (0.6%)
    - desk: 1972 (0.6%)
    - case of water bottles: 1780 (0.5%)
    - box: 1198 (0.3%)

## desk (idx=8)
- gt=655163 pred=637674 IoU=0.4305 recall=0.5939 precision=0.6101 cost=266087
- FN total=266087, top-1 destination share=11.3%
    - dresser: 30135 (11.3%)
    - table: 25884 (9.7%)
    - bench: 23284 (8.8%)
    - paper: 17177 (6.5%)
    - shelf: 16179 (6.1%)
    - keyboard: 14896 (5.6%)
    - cabinet: 13082 (4.9%)
    - monitor: 12890 (4.8%)

## trash can (idx=48)
- gt=298994 pred=56975 IoU=0.1429 recall=0.1488 precision=0.7811 cost=254491
- FN total=254491, top-1 destination share=46.2%
    - trash bin: 117475 (46.2%)
    - bucket: 22285 (8.8%)
    - water cooler: 19564 (7.7%)
    - recycling bin: 16888 (6.6%)
    - laundry hamper: 9020 (3.5%)
    - laundry basket: 9017 (3.5%)
    - storage bin: 5727 (2.3%)
    - paper bag: 4475 (1.8%)

## mailbox (idx=163)
- gt=217967 pred=0 IoU=0.0000 recall=0.0000 precision=0.0000 cost=217967
- FN total=217967, top-1 destination share=35.8%
    - bookshelf: 77927 (35.8%)
    - shelf: 55699 (25.6%)
    - bathroom stall: 17738 (8.1%)
    - refrigerator: 10523 (4.8%)
    - bulletin board: 9404 (4.3%)
    - cabinet: 7369 (3.4%)
    - file cabinet: 7078 (3.2%)
    - stair rail: 7072 (3.2%)

## curtain (idx=18)
- gt=657363 pred=523346 IoU=0.6211 recall=0.6881 precision=0.8643 cost=205013
- FN total=205013, top-1 destination share=30.5%
    - window: 62440 (30.5%)
    - blinds: 46723 (22.8%)
    - shower curtain: 19855 (9.7%)
    - decoration: 14637 (7.1%)
    - closet wall: 8358 (4.1%)
    - bulletin board: 6768 (3.3%)
    - windowsill: 5278 (2.6%)
    - plant: 4142 (2.0%)

## bookshelf (idx=16)
- gt=1125319 pred=1808296 IoU=0.4577 recall=0.8185 precision=0.5094 cost=204248
- FN total=204249, top-1 destination share=49.8%
    - shelf: 101747 (49.8%)
    - storage organizer: 17901 (8.8%)
    - window: 10484 (5.1%)
    - computer tower: 9219 (4.5%)
    - table: 8959 (4.4%)
    - rack: 6998 (3.4%)
    - file cabinet: 4949 (2.4%)
    - closet wall: 4383 (2.1%)

## clothes dryer (idx=92)
- gt=206786 pred=92592 IoU=0.0594 recall=0.0812 precision=0.1813 cost=190001
- FN total=190001, top-1 destination share=46.8%
    - oven: 88987 (46.8%)
    - stove: 32480 (17.1%)
    - washing machine: 25521 (13.4%)
    - refrigerator: 8130 (4.3%)
    - shower: 4610 (2.4%)
    - range hood: 4497 (2.4%)
    - rail: 4342 (2.3%)
    - water cooler: 3593 (1.9%)

## whiteboard (idx=45)
- gt=323582 pred=219672 IoU=0.3465 recall=0.4321 precision=0.6364 cost=183777
- FN total=183777, top-1 destination share=76.9%
    - board: 141390 (76.9%)
    - bulletin board: 17736 (9.7%)
    - poster: 4923 (2.7%)
    - blackboard: 4220 (2.3%)
    - refrigerator: 2747 (1.5%)
    - window: 1534 (0.8%)
    - office chair: 1523 (0.8%)
    - bookshelf: 1505 (0.8%)

## doorframe (idx=118)
- gt=185363 pred=48464 IoU=0.0464 recall=0.0559 precision=0.2137 cost=175005
- FN total=175005, top-1 destination share=58.3%
    - door: 102090 (58.3%)
    - shower door: 19693 (11.3%)
    - closet door: 10970 (6.3%)
    - bathroom stall door: 10962 (6.3%)
    - closet wall: 7670 (4.4%)
    - closet: 6667 (3.8%)
    - shower wall: 2233 (1.3%)
    - shower curtain: 2211 (1.3%)

## armchair (idx=20)
- gt=320557 pred=281916 IoU=0.3910 recall=0.5283 precision=0.6007 cost=151202
- FN total=151202, top-1 destination share=43.5%
    - sofa chair: 65774 (43.5%)
    - seat: 36009 (23.8%)
    - chair: 12297 (8.1%)
    - couch: 11835 (7.8%)
    - mini fridge: 3820 (2.5%)
    - file cabinet: 2854 (1.9%)
    - door: 2746 (1.8%)
    - ottoman: 2111 (1.4%)

## pillow (idx=11)
- gt=153508 pred=25721 IoU=0.0957 recall=0.1020 precision=0.6088 cost=137850
- FN total=137850, top-1 destination share=62.2%
    - bed: 85793 (62.2%)
    - couch: 22836 (16.6%)
    - ottoman: 4037 (2.9%)
    - blanket: 2645 (1.9%)
    - clothes: 2024 (1.5%)
    - armchair: 1743 (1.3%)
    - picture: 1703 (1.2%)
    - table: 1689 (1.2%)

## box (idx=22)
- gt=178867 pred=82276 IoU=0.2116 recall=0.2550 precision=0.5544 cost=133254
- FN total=133254, top-1 destination share=8.0%
    - crate: 10721 (8.0%)
    - storage bin: 10585 (7.9%)
    - laundry hamper: 9484 (7.1%)
    - printer: 8531 (6.4%)
    - bookshelf: 5734 (4.3%)
    - storage organizer: 5514 (4.1%)
    - paper bag: 5018 (3.8%)
    - shelf: 4998 (3.8%)

## bed (idx=10)
- gt=841283 pred=854700 IoU=0.7266 recall=0.8484 precision=0.8350 cost=127570
- FN total=127570, top-1 destination share=16.0%
    - mattress: 20387 (16.0%)
    - blanket: 14903 (11.7%)
    - nightstand: 11832 (9.3%)
    - desk: 10770 (8.4%)
    - bookshelf: 10683 (8.4%)
    - dresser: 8877 (7.0%)
    - clothes: 4746 (3.7%)
    - mat: 4602 (3.6%)

## object (idx=172)
- gt=125030 pred=1 IoU=0.0000 recall=0.0000 precision=0.0000 cost=125030
- FN total=125030, top-1 destination share=10.7%
    - bicycle: 13436 (10.7%)
    - guitar case: 7516 (6.0%)
    - suitcase: 5136 (4.1%)
    - window: 4986 (4.0%)
    - box: 4955 (4.0%)
    - fireplace: 4614 (3.7%)
    - paper bag: 4220 (3.4%)
    - folded chair: 4033 (3.2%)

## picture (idx=13)
- gt=184537 pred=82559 IoU=0.3013 recall=0.3351 precision=0.7491 cost=122695
- FN total=122696, top-1 destination share=27.8%
    - poster: 34058 (27.8%)
    - bulletin board: 28411 (23.2%)
    - decoration: 13779 (11.2%)
    - door: 12603 (10.3%)
    - blackboard: 3969 (3.2%)
    - closet wall: 2895 (2.4%)
    - bar: 2228 (1.8%)
    - bookshelf: 1993 (1.6%)

## couch (idx=5)
- gt=789769 pred=774737 IoU=0.7610 recall=0.8560 precision=0.8726 cost=113702
- FN total=113702, top-1 destination share=51.7%
    - bench: 58788 (51.7%)
    - ottoman: 12718 (11.2%)
    - armchair: 9482 (8.3%)
    - furniture: 4776 (4.2%)
    - blanket: 3521 (3.1%)
    - clothes: 3514 (3.1%)
    - pillow: 2730 (2.4%)
    - window: 2565 (2.3%)

## blinds (idx=73)
- gt=150500 pred=308447 IoU=0.1024 recall=0.2833 precision=0.1382 cost=107859
- FN total=107859, top-1 destination share=22.1%
    - closet wall: 23794 (22.1%)
    - window: 20944 (19.4%)
    - curtain: 20447 (19.0%)
    - bulletin board: 14575 (13.5%)
    - bookshelf: 8054 (7.5%)
    - blackboard: 5301 (4.9%)
    - closet door: 3253 (3.0%)
    - decoration: 3132 (2.9%)

## copier (idx=62)
- gt=99703 pred=7670 IoU=0.0428 recall=0.0442 precision=0.5750 cost=95293
- FN total=95293, top-1 destination share=41.3%
    - printer: 39344 (41.3%)
    - mini fridge: 20061 (21.1%)
    - washing machine: 12768 (13.4%)
    - computer tower: 7154 (7.5%)
    - trash bin: 3032 (3.2%)
    - file cabinet: 2814 (3.0%)
    - water cooler: 2679 (2.8%)
    - clothes dryer: 1251 (1.3%)

## radiator (idx=80)
- gt=96438 pred=4962 IoU=0.0355 recall=0.0360 precision=0.6999 cost=92965
- FN total=92965, top-1 destination share=45.0%
    - vent: 41820 (45.0%)
    - closet wall: 8141 (8.8%)
    - windowsill: 5874 (6.3%)
    - window: 3855 (4.1%)
    - refrigerator: 2882 (3.1%)
    - blackboard: 2300 (2.5%)
    - laundry hamper: 2176 (2.3%)
    - couch: 2115 (2.3%)

