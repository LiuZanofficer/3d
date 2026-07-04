# Task 0 real 200-class oracle ceiling

## Self-check
- baseline fg-mIoU = **0.1548** (ref 0.1548; OK)
- flip threshold = 0.5
- scoring: pred_semantic, background->ignore, invalid labels dropped, fg_class_idx mean

## PRE-REGISTERED criteria
- <=0.18: low ceiling / warn
- >=0.21: physical room for 20+ fg-mIoU
- otherwise: modest room

## Oracle results

| top-k victim classes | n_classes | oracle fg-mIoU | gain over baseline | verdict |
|---|---:|---:|---:|---|
| 25 | 25 | 0.1906 | +0.0358 | MODEST_ROOM (0.18,0.21): continue, but lower expectations |
| 50 | 50 | 0.2118 | +0.0570 | ROOM_FOR_20_PLUS (>=0.21): physical room exists |
| all | 186 | 0.3257 | +0.1709 | ROOM_FOR_20_PLUS (>=0.21): physical room exists |

## Per-class contribution (25)

| C | absorber J | cost | corrected_inst | corrected_points | base_iou | oracle_iou | delta_iou |
|---|---|---:|---:|---:|---:|---:|---:|
| chair | office chair | 1237268 | 311 | 668697 | 0.4915 | 0.7367 | +0.2452 |
| shelf | bookshelf | 810674 | 41 | 400477 | 0.2499 | 0.5222 | +0.2723 |
| cabinet | kitchen cabinet | 624692 | 22 | 81018 | 0.1603 | 0.2529 | +0.0926 |
| window | blinds | 561323 | 18 | 187964 | 0.4969 | 0.6341 | +0.1372 |
| door | window | 451372 | 7 | 101026 | 0.5281 | 0.6245 | +0.0964 |
| table | desk | 373151 | 25 | 106794 | 0.5898 | 0.6604 | +0.0706 |
| book | bookshelf | 352139 | 364 | 320569 | 0.0001 | 0.9103 | +0.9101 |
| desk | dresser | 266085 | 2 | 12099 | 0.4305 | 0.4837 | +0.0531 |
| trash can | trash bin | 254491 | 83 | 127241 | 0.1429 | 0.5147 | +0.3718 |
| mailbox | bookshelf | 217967 | 2 | 85597 | 0.0000 | 0.3927 | +0.3927 |
| curtain | window | 205013 | 7 | 32802 | 0.6211 | 0.6613 | +0.0402 |
| bookshelf | shelf | 204247 | 10 | 109625 | 0.4577 | 0.7723 | +0.3146 |
| clothes dryer | oven | 190001 | 3 | 81356 | 0.0594 | 0.3441 | +0.2847 |
| whiteboard | board | 183777 | 27 | 147717 | 0.3465 | 0.6601 | +0.3136 |
| doorframe | door | 175002 | 73 | 108810 | 0.0464 | 0.5095 | +0.4632 |
| armchair | sofa chair | 151202 | 13 | 57274 | 0.3910 | 0.5150 | +0.1239 |
| pillow | bed | 137850 | 123 | 90669 | 0.0957 | 0.6173 | +0.5216 |
| box | crate | 133254 | 2 | 6274 | 0.2116 | 0.2427 | +0.0311 |
| bed | mattress | 127571 | 0 | 0 | 0.7266 | 0.7965 | +0.0699 |
| object | bicycle | 125030 | 3 | 14267 | 0.0000 | 0.1141 | +0.1141 |
| picture | poster | 122695 | 69 | 34487 | 0.3013 | 0.4593 | +0.1580 |
| couch | bench | 113702 | 7 | 54065 | 0.7610 | 0.8043 | +0.0433 |
| blinds | closet wall | 107859 | 4 | 27741 | 0.1024 | 0.2521 | +0.1497 |
| copier | printer | 95293 | 5 | 32885 | 0.0428 | 0.3339 | +0.2911 |
| radiator | vent | 92965 | 24 | 47803 | 0.0355 | 0.5049 | +0.4694 |

## Per-class contribution (50)

| C | absorber J | cost | corrected_inst | corrected_points | base_iou | oracle_iou | delta_iou |
|---|---|---:|---:|---:|---:|---:|---:|
| chair | office chair | 1237268 | 311 | 668697 | 0.4915 | 0.7368 | +0.2453 |
| shelf | bookshelf | 810674 | 41 | 400477 | 0.2499 | 0.5222 | +0.2723 |
| cabinet | kitchen cabinet | 624692 | 22 | 81018 | 0.1603 | 0.2529 | +0.0926 |
| window | blinds | 561323 | 18 | 187964 | 0.4969 | 0.6342 | +0.1374 |
| door | window | 451372 | 7 | 101026 | 0.5281 | 0.6259 | +0.0978 |
| table | desk | 373151 | 25 | 106794 | 0.5898 | 0.6608 | +0.0710 |
| book | bookshelf | 352139 | 364 | 320569 | 0.0001 | 0.9103 | +0.9101 |
| desk | dresser | 266085 | 2 | 12099 | 0.4305 | 0.4901 | +0.0595 |
| trash can | trash bin | 254491 | 83 | 127241 | 0.1429 | 0.5147 | +0.3718 |
| mailbox | bookshelf | 217967 | 2 | 85597 | 0.0000 | 0.3927 | +0.3927 |
| curtain | window | 205013 | 7 | 32802 | 0.6211 | 0.6628 | +0.0418 |
| bookshelf | shelf | 204247 | 10 | 109625 | 0.4577 | 0.7723 | +0.3146 |
| clothes dryer | oven | 190001 | 3 | 81356 | 0.0594 | 0.3441 | +0.2847 |
| whiteboard | board | 183777 | 27 | 147717 | 0.3465 | 0.6838 | +0.3373 |
| doorframe | door | 175002 | 73 | 108810 | 0.0464 | 0.5140 | +0.4677 |
| armchair | sofa chair | 151202 | 13 | 57274 | 0.3910 | 0.5919 | +0.2009 |
| pillow | bed | 137850 | 123 | 90669 | 0.0957 | 0.6224 | +0.5267 |
| box | crate | 133254 | 2 | 6274 | 0.2116 | 0.2427 | +0.0311 |
| bed | mattress | 127571 | 0 | 0 | 0.7266 | 0.8179 | +0.0913 |
| object | bicycle | 125030 | 3 | 14267 | 0.0000 | 0.1141 | +0.1141 |
| picture | poster | 122695 | 69 | 34487 | 0.3013 | 0.4594 | +0.1581 |
| couch | bench | 113702 | 7 | 54065 | 0.7610 | 0.8100 | +0.0491 |
| blinds | closet wall | 107859 | 4 | 27741 | 0.1024 | 0.2521 | +0.1497 |
| copier | printer | 95293 | 5 | 32885 | 0.0428 | 0.3343 | +0.2915 |
| radiator | vent | 92965 | 24 | 47803 | 0.0355 | 0.5049 | +0.4694 |
| blackboard | board | 87329 | 11 | 80572 | 0.2314 | 0.6006 | +0.3692 |
| shower wall | shower | 87206 | 0 | 0 | 0.4957 | 0.4965 | +0.0007 |
| backpack | bag | 85330 | 9 | 10773 | 0.3858 | 0.4481 | +0.0623 |
| sofa chair | armchair | 83213 | 13 | 65902 | 0.0141 | 0.5252 | +0.5111 |
| shower | shower door | 77957 | 2 | 33583 | 0.0592 | 0.2913 | +0.2320 |
| bathroom vanity | bathroom counter | 70308 | 9 | 29253 | 0.1216 | 0.4193 | +0.2977 |
| coffee table | ottoman | 67373 | 7 | 31216 | 0.4212 | 0.5555 | +0.1343 |
| mattress | clothes | 67227 | 1 | 10096 | 0.0000 | 0.1134 | +0.1134 |
| cart | storage organizer | 65801 | 3 | 9603 | 0.1307 | 0.2469 | +0.1162 |
| stair rail | clothes | 59391 | 0 | 0 | 0.1955 | 0.1956 | +0.0001 |
| kitchen cabinet | kitchen counter | 58966 | 0 | 0 | 0.6797 | 0.7599 | +0.0802 |
| washing machine | clothes dryer | 57604 | 0 | 0 | 0.6643 | 0.6791 | +0.0147 |
| wardrobe | closet | 56087 | 3 | 11994 | 0.1681 | 0.2742 | +0.1061 |
| clothes | closet | 53298 | 5 | 9236 | 0.1265 | 0.2103 | +0.0838 |
| counter | kitchen counter | 46974 | 4 | 23719 | 0.2041 | 0.3311 | +0.1270 |
| bathroom stall | bathroom stall door | 46383 | 0 | 0 | 0.4996 | 0.5040 | +0.0045 |
| blanket | bed | 46337 | 7 | 28199 | 0.0556 | 0.3809 | +0.3253 |
| mini fridge | suitcase | 45608 | 3 | 11470 | 0.0032 | 0.1329 | +0.1297 |
| projector screen | whiteboard | 45457 | 2 | 16666 | 0.0952 | 0.3858 | +0.2906 |
| file cabinet | desk | 43522 | 17 | 14049 | 0.2228 | 0.2991 | +0.0763 |
| seat | bench | 43149 | 2 | 13723 | 0.0108 | 0.0949 | +0.0841 |
| closet door | closet wall | 41760 | 6 | 21190 | 0.0973 | 0.2078 | +0.1105 |
| pillar | column | 41664 | 2 | 23054 | 0.1612 | 0.5149 | +0.3537 |
| closet | closet door | 41549 | 3 | 15135 | 0.0587 | 0.1216 | +0.0629 |
| bag | basket | 41501 | 5 | 6280 | 0.0812 | 0.1841 | +0.1029 |

## Per-class contribution (all)

| C | absorber J | cost | corrected_inst | corrected_points | base_iou | oracle_iou | delta_iou |
|---|---|---:|---:|---:|---:|---:|---:|
| chair | office chair | 1237268 | 311 | 668697 | 0.4915 | 0.7427 | +0.2512 |
| shelf | bookshelf | 810674 | 41 | 400477 | 0.2499 | 0.5260 | +0.2761 |
| cabinet | kitchen cabinet | 624692 | 22 | 81018 | 0.1603 | 0.2529 | +0.0926 |
| window | blinds | 561323 | 18 | 187964 | 0.4969 | 0.6456 | +0.1487 |
| door | window | 451372 | 7 | 101026 | 0.5281 | 0.6263 | +0.0981 |
| table | desk | 373151 | 25 | 106794 | 0.5898 | 0.6821 | +0.0923 |
| book | bookshelf | 352139 | 364 | 320569 | 0.0001 | 0.9103 | +0.9101 |
| desk | dresser | 266085 | 2 | 12099 | 0.4305 | 0.4932 | +0.0627 |
| trash can | trash bin | 254491 | 83 | 127241 | 0.1429 | 0.5177 | +0.3748 |
| mailbox | bookshelf | 217967 | 2 | 85597 | 0.0000 | 0.3927 | +0.3927 |
| curtain | window | 205013 | 7 | 32802 | 0.6211 | 0.6648 | +0.0438 |
| bookshelf | shelf | 204247 | 10 | 109625 | 0.4577 | 0.7897 | +0.3320 |
| clothes dryer | oven | 190001 | 3 | 81356 | 0.0594 | 0.3520 | +0.2926 |
| whiteboard | board | 183777 | 27 | 147717 | 0.3465 | 0.6933 | +0.3468 |
| doorframe | door | 175002 | 73 | 108810 | 0.0464 | 0.5145 | +0.4682 |
| armchair | sofa chair | 151202 | 13 | 57274 | 0.3910 | 0.5923 | +0.2012 |
| pillow | bed | 137850 | 123 | 90669 | 0.0957 | 0.6227 | +0.5269 |
| box | crate | 133254 | 2 | 6274 | 0.2116 | 0.2447 | +0.0331 |
| bed | mattress | 127571 | 0 | 0 | 0.7266 | 0.8189 | +0.0923 |
| object | bicycle | 125030 | 3 | 14267 | 0.0000 | 0.1141 | +0.1141 |
| picture | poster | 122695 | 69 | 34487 | 0.3013 | 0.4814 | +0.1801 |
| couch | bench | 113702 | 7 | 54065 | 0.7610 | 0.8134 | +0.0525 |
| blinds | closet wall | 107859 | 4 | 27741 | 0.1024 | 0.2527 | +0.1503 |
| copier | printer | 95293 | 5 | 32885 | 0.0428 | 0.3343 | +0.2915 |
| radiator | vent | 92965 | 24 | 47803 | 0.0355 | 0.5049 | +0.4694 |
| blackboard | board | 87329 | 11 | 80572 | 0.2314 | 0.6221 | +0.3907 |
| shower wall | shower | 87206 | 0 | 0 | 0.4957 | 0.4997 | +0.0040 |
| backpack | bag | 85330 | 9 | 10773 | 0.3858 | 0.4481 | +0.0623 |
| sofa chair | armchair | 83213 | 13 | 65902 | 0.0141 | 0.5278 | +0.5137 |
| shower | shower door | 77957 | 2 | 33583 | 0.0592 | 0.2931 | +0.2338 |
| bathroom vanity | bathroom counter | 70308 | 9 | 29253 | 0.1216 | 0.4198 | +0.2982 |
| coffee table | ottoman | 67373 | 7 | 31216 | 0.4212 | 0.5673 | +0.1462 |
| mattress | clothes | 67227 | 1 | 10096 | 0.0000 | 0.1134 | +0.1134 |
| cart | storage organizer | 65801 | 3 | 9603 | 0.1307 | 0.2469 | +0.1162 |
| stair rail | clothes | 59391 | 0 | 0 | 0.1955 | 0.2354 | +0.0399 |
| kitchen cabinet | kitchen counter | 58966 | 0 | 0 | 0.6797 | 0.7730 | +0.0934 |
| washing machine | clothes dryer | 57604 | 0 | 0 | 0.6643 | 0.6800 | +0.0157 |
| wardrobe | closet | 56087 | 3 | 11994 | 0.1681 | 0.2808 | +0.1126 |
| clothes | closet | 53298 | 5 | 9236 | 0.1265 | 0.2282 | +0.1017 |
| counter | kitchen counter | 46974 | 4 | 23719 | 0.2041 | 0.3475 | +0.1434 |
| bathroom stall | bathroom stall door | 46383 | 0 | 0 | 0.4996 | 0.5304 | +0.0309 |
| blanket | bed | 46337 | 7 | 28199 | 0.0556 | 0.3809 | +0.3253 |
| mini fridge | suitcase | 45608 | 3 | 11470 | 0.0032 | 0.1330 | +0.1298 |
| projector screen | whiteboard | 45457 | 2 | 16666 | 0.0952 | 0.3859 | +0.2907 |
| file cabinet | desk | 43522 | 17 | 14049 | 0.2228 | 0.2993 | +0.0765 |
| seat | bench | 43149 | 2 | 13723 | 0.0108 | 0.0973 | +0.0865 |
| closet door | closet wall | 41760 | 6 | 21190 | 0.0973 | 0.2079 | +0.1106 |
| pillar | column | 41664 | 2 | 23054 | 0.1612 | 0.5457 | +0.3845 |
| closet | closet door | 41549 | 3 | 15135 | 0.0587 | 0.1297 | +0.0710 |
| bag | basket | 41501 | 5 | 6280 | 0.0812 | 0.1841 | +0.1029 |
| refrigerator | window | 41417 | 2 | 10838 | 0.4949 | 0.5479 | +0.0530 |
| dresser | closet | 40563 | 4 | 13042 | 0.3288 | 0.3993 | +0.0705 |
| end table | table | 40185 | 5 | 12556 | 0.0503 | 0.2130 | +0.1627 |
| office chair | chair | 37888 | 13 | 24022 | 0.2438 | 0.6445 | +0.4008 |
| stove | oven | 37634 | 9 | 32611 | 0.2592 | 0.5811 | +0.3219 |
| mirror | bathroom counter | 37565 | 2 | 2230 | 0.3527 | 0.3788 | +0.0261 |
| shower curtain | shower door | 36671 | 2 | 7121 | 0.6049 | 0.6725 | +0.0675 |
| recycling bin | trash bin | 34670 | 15 | 20952 | 0.1622 | 0.4364 | +0.2742 |
| board | whiteboard | 34454 | 4 | 3463 | 0.0388 | 0.1167 | +0.0779 |
| kitchen counter | counter | 33787 | 4 | 12300 | 0.4753 | 0.6017 | +0.1264 |
| windowsill | window | 30818 | 8 | 20846 | 0.0872 | 0.2776 | +0.1905 |
| rail | stair rail | 30757 | 2 | 17247 | 0.0043 | 0.4365 | +0.4322 |
| divider | board | 29765 | 1 | 7346 | 0.0000 | 0.2376 | +0.2376 |
| ottoman | seat | 29377 | 4 | 5571 | 0.1019 | 0.1821 | +0.0803 |
| monitor | computer tower | 28899 | 3 | 2803 | 0.7745 | 0.8053 | +0.0308 |
| towel | shower curtain | 28789 | 11 | 10194 | 0.4390 | 0.5816 | +0.1426 |
| dining table | table | 27674 | 3 | 32397 | 0.0438 | 0.3007 | +0.2569 |
| sink | kitchen counter | 26120 | 3 | 3112 | 0.5597 | 0.6182 | +0.0585 |
| furniture | bookshelf | 25829 | 1 | 21272 | 0.0000 | 0.4251 | +0.4251 |
| jacket | clothes | 23408 | 3 | 8044 | 0.0337 | 0.1998 | +0.1660 |
| bench | table | 22243 | 3 | 5194 | 0.1658 | 0.2422 | +0.0764 |
| suitcase | luggage | 22134 | 6 | 10313 | 0.1858 | 0.3026 | +0.1168 |
| shoe | vacuum cleaner | 22047 | 11 | 3179 | 0.2157 | 0.3095 | +0.0938 |
| light | stair rail | 21673 | 1 | 14023 | 0.0050 | 0.6366 | +0.6316 |
| computer tower | desk | 21643 | 4 | 3607 | 0.1777 | 0.2199 | +0.0422 |
| poster | picture | 21145 | 3 | 10936 | 0.0217 | 0.2348 | +0.2131 |
| closet wall | closet | 20446 | 1 | 9422 | 0.0127 | 0.0591 | +0.0464 |
| rack | bookshelf | 19529 | 1 | 8452 | 0.0006 | 0.0697 | +0.0692 |
| ledge | shelf | 18949 | 5 | 6315 | 0.0000 | 0.2778 | +0.2778 |
| laundry hamper | bucket | 18340 | 0 | 0 | 0.0224 | 0.0234 | +0.0010 |
| bathtub | shower floor | 18313 | 3 | 2858 | 0.7551 | 0.7901 | +0.0350 |
| piano | keyboard piano | 18251 | 0 | 0 | 0.2288 | 0.2288 | +0.0000 |
| laundry basket | clothes dryer | 18100 | 1 | 3027 | 0.0409 | 0.1067 | +0.0658 |
| lamp | curtain | 17112 | 3 | 2956 | 0.5923 | 0.6246 | +0.0323 |
| fan | clothes dryer | 16550 | 3 | 4855 | 0.1240 | 0.2757 | +0.1517 |
| bathroom stall door | bathroom stall | 16537 | 6 | 12619 | 0.1192 | 0.2095 | +0.0904 |
| tv | blackboard | 16303 | 2 | 6927 | 0.7613 | 0.8127 | +0.0514 |
| trash bin | water cooler | 16020 | 1 | 3449 | 0.0581 | 0.1880 | +0.1300 |
| machine | refrigerator | 15411 | 0 | 0 | 0.0855 | 0.0859 | +0.0004 |
| nightstand | dresser | 15402 | 2 | 3286 | 0.6124 | 0.6665 | +0.0541 |
| printer | toaster oven | 14480 | 2 | 3249 | 0.1838 | 0.2740 | +0.0902 |
| storage bin | nightstand | 14281 | 3 | 2658 | 0.0137 | 0.0643 | +0.0506 |
| microwave | toaster oven | 13679 | 5 | 6938 | 0.2830 | 0.4767 | +0.1938 |
| case of water bottles | vacuum cleaner | 13047 | 1 | 1595 | 0.0277 | 0.0534 | +0.0257 |
| ladder | rack | 11706 | 2 | 6757 | 0.2214 | 0.4426 | +0.2212 |
| plant | case of water bottles | 11166 | 3 | 1445 | 0.7884 | 0.8321 | +0.0437 |
| pipe | vent | 11043 | 1 | 2083 | 0.0000 | 0.1842 | +0.1842 |
| stool | toilet | 10873 | 1 | 1821 | 0.1587 | 0.1743 | +0.0156 |
| vacuum cleaner | plant | 10704 | 2 | 4531 | 0.0789 | 0.2107 | +0.1318 |
| bin | storage bin | 10633 | 1 | 1154 | 0.0001 | 0.0949 | +0.0947 |
| ironing board | shower door | 10331 | 2 | 2874 | 0.0000 | 0.2357 | +0.2357 |
| mat | towel | 10025 | 5 | 4105 | 0.2764 | 0.3851 | +0.1088 |
| bathroom cabinet | bathroom counter | 9834 | 0 | 0 | 0.2159 | 0.2197 | +0.0038 |
| dishwasher | kitchen cabinet | 9705 | 5 | 7495 | 0.1579 | 0.4030 | +0.2451 |
| telephone | laptop | 9642 | 6 | 1742 | 0.0553 | 0.1569 | +0.1016 |
| paper | bookshelf | 9280 | 5 | 2050 | 0.0520 | 0.1056 | +0.0536 |
| paper towel dispenser | toilet paper dispenser | 9146 | 3 | 2991 | 0.3163 | 0.4241 | +0.1078 |
| stairs | stair rail | 8684 | 0 | 0 | 0.4930 | 0.5008 | +0.0079 |
| column | pillar | 8449 | 1 | 4570 | 0.1655 | 0.4526 | +0.2870 |
| laptop | monitor | 8254 | 4 | 3730 | 0.1807 | 0.2726 | +0.0919 |
| person | couch | 8223 | 3 | 4146 | 0.1098 | 0.3601 | +0.2503 |
| clock | decoration | 7488 | 3 | 2276 | 0.1662 | 0.3960 | +0.2298 |
| tissue box | case of water bottles | 7378 | 2 | 827 | 0.0290 | 0.1060 | +0.0770 |
| oven | refrigerator | 7247 | 4 | 3012 | 0.0568 | 0.1348 | +0.0780 |
| toilet paper | toilet paper holder | 7129 | 12 | 2235 | 0.2602 | 0.3879 | +0.1277 |
| crate | stool | 6964 | 3 | 4557 | 0.0000 | 0.1188 | +0.1188 |
| bulletin board | poster | 6909 | 1 | 1563 | 0.1259 | 0.1450 | +0.0190 |
| bathroom counter | sink | 6787 | 1 | 656 | 0.0719 | 0.1086 | +0.0367 |
| keyboard piano | plant | 6705 | 1 | 1788 | 0.0167 | 0.0797 | +0.0630 |
| toilet seat cover dispenser | paper towel dispenser | 6587 | 6 | 3184 | 0.0038 | 0.4725 | +0.4687 |
| container | laundry basket | 6391 | 5 | 3810 | 0.0000 | 0.5318 | +0.5318 |
| tray | table | 5920 | 2 | 1359 | 0.0000 | 0.2278 | +0.2278 |
| water cooler | storage bin | 5856 | 0 | 0 | 0.0023 | 0.0026 | +0.0003 |
| shower door | shower curtain | 5799 | 0 | 0 | 0.0358 | 0.0473 | +0.0115 |
| bucket | laundry basket | 5770 | 4 | 2566 | 0.0475 | 0.1034 | +0.0559 |
| toilet | towel | 5652 | 0 | 0 | 0.8418 | 0.8581 | +0.0163 |
| soap dispenser | paper towel dispenser | 5623 | 14 | 4432 | 0.0615 | 0.4014 | +0.3399 |
| bottle | case of water bottles | 5226 | 5 | 994 | 0.0032 | 0.1879 | +0.1848 |
| paper cutter | table | 5115 | 2 | 1863 | 0.0189 | 0.2691 | +0.2502 |
| sign | shower wall | 4912 | 1 | 821 | 0.0000 | 0.1669 | +0.1669 |
| tube | coffee table | 4879 | 5 | 2206 | 0.0000 | 0.3226 | +0.3226 |
| keyboard | desk | 4858 | 5 | 569 | 0.2105 | 0.2512 | +0.0407 |
| toilet paper dispenser | towel | 4814 | 2 | 1762 | 0.2295 | 0.3574 | +0.1280 |
| decoration | poster | 4594 | 2 | 2569 | 0.0308 | 0.0618 | +0.0310 |
| tv stand | dresser | 4438 | 0 | 0 | 0.3761 | 0.3799 | +0.0038 |
| cup | coffee kettle | 4271 | 9 | 1311 | 0.0346 | 0.2483 | +0.2137 |
| bar | bathroom stall | 4044 | 2 | 915 | 0.0000 | 0.0317 | +0.0317 |
| basket | crate | 3780 | 2 | 3048 | 0.0005 | 0.2188 | +0.2183 |
| range hood | kitchen cabinet | 3609 | 3 | 2514 | 0.2828 | 0.4810 | +0.1982 |
| guitar | fan | 3535 | 2 | 2220 | 0.0000 | 0.3157 | +0.3157 |
| dish rack | microwave | 3457 | 2 | 1776 | 0.0115 | 0.0896 | +0.0781 |
| paper bag | box | 3379 | 1 | 934 | 0.0333 | 0.0547 | +0.0215 |
| paper towel roll | tissue box | 3376 | 1 | 624 | 0.0871 | 0.2229 | +0.1358 |
| toaster oven | microwave | 3149 | 1 | 1576 | 0.1146 | 0.1966 | +0.0820 |
| laundry detergent | case of water bottles | 2938 | 4 | 1680 | 0.0827 | 0.4683 | +0.3856 |
| broom | vacuum cleaner | 2714 | 2 | 1203 | 0.0000 | 0.3680 | +0.3680 |
| soap dish | shower wall | 2357 | 6 | 625 | 0.0442 | 0.2921 | +0.2479 |
| fire extinguisher | bar | 2190 | 1 | 1125 | 0.0048 | 0.2779 | +0.2731 |
| bowl | case of water bottles | 2014 | 1 | 526 | 0.0000 | 0.2283 | +0.2283 |
| ceiling light | vent | 1818 | 0 | 0 | 0.2622 | 0.3048 | +0.0427 |
| potted plant | plant | 1792 | 1 | 1792 | 0.0000 | 0.3854 | +0.3854 |
| fireplace | couch | 1507 | 0 | 0 | 0.5192 | 0.5291 | +0.0098 |
| shower head | mirror | 1351 | 1 | 625 | 0.1207 | 0.4979 | +0.3772 |
| plate | desk | 1272 | 4 | 582 | 0.0000 | 0.4110 | +0.4110 |
| shower curtain rod | shower curtain | 1230 | 2 | 678 | 0.0599 | 0.1080 | +0.0482 |
| cushion | blanket | 1192 | 0 | 0 | 0.0000 | 0.0000 | +0.0000 |
| hat | fan | 1172 | 1 | 559 | 0.0000 | 0.4222 | +0.4222 |
| hair dryer | mirror | 1008 | 2 | 1008 | 0.0000 | 0.9456 | +0.9456 |
| headphones | office chair | 1002 | 2 | 405 | 0.0000 | 0.4042 | +0.4042 |
| folded chair | chair | 949 | 0 | 0 | 0.0222 | 0.0231 | +0.0010 |
| calendar | bookshelf | 895 | 2 | 601 | 0.0000 | 0.6547 | +0.6547 |
| stand | fan | 874 | 2 | 874 | 0.0000 | 0.2338 | +0.2338 |
| toaster | toaster oven | 801 | 4 | 801 | 0.0000 | 0.1580 | +0.1580 |
| closet rod | closet | 794 | 1 | 435 | 0.0394 | 0.0969 | +0.0576 |
| water bottle | water pitcher | 782 | 2 | 362 | 0.0000 | 0.0990 | +0.0990 |
| coffee maker | water cooler | 774 | 0 | 0 | 0.3059 | 0.3457 | +0.0398 |
| vent | ceiling light | 770 | 2 | 585 | 0.0000 | 0.0208 | +0.0208 |
| dustpan | toilet | 762 | 0 | 0 | 0.0000 | 0.0000 | +0.0000 |
| power outlet | closet wall | 711 | 3 | 377 | 0.0000 | 0.5302 | +0.5302 |
| scale | mat | 665 | 0 | 0 | 0.0000 | 0.0000 | +0.0000 |
| plunger | toilet | 613 | 2 | 613 | 0.0000 | 1.0000 | +1.0000 |
| handicap bar | bathroom stall | 598 | 2 | 599 | 0.0007 | 0.3900 | +0.3893 |
| water pitcher | coffee maker | 579 | 1 | 579 | 0.0000 | 0.3878 | +0.3878 |
| dumbbell | office chair | 507 | 1 | 327 | 0.0000 | 0.6450 | +0.6450 |
| coffee kettle | coffee maker | 491 | 1 | 173 | 0.0236 | 0.0438 | +0.0202 |
| ball | vacuum cleaner | 451 | 0 | 0 | 0.2422 | 0.2693 | +0.0272 |
| light switch | shower door | 432 | 3 | 155 | 0.0000 | 0.3588 | +0.3588 |
| coat rack | guitar | 416 | 0 | 0 | 0.0087 | 0.0094 | +0.0008 |
| mouse | desk | 350 | 4 | 184 | 0.0000 | 0.5257 | +0.5257 |
| speaker | monitor | 347 | 2 | 348 | 0.0001 | 0.0336 | +0.0335 |
| projector | ceiling light | 283 | 1 | 283 | 0.0000 | 0.1643 | +0.1643 |
| toilet paper holder | shower wall | 253 | 0 | 0 | 0.0531 | 0.0633 | +0.0103 |
| stuffed animal | bed | 228 | 1 | 228 | 0.0000 | 0.0482 | +0.0482 |
| fire alarm | paper towel dispenser | 61 | 1 | 61 | 0.0000 | 0.0484 | +0.0484 |
| shower floor | shower door | 52 | 0 | 0 | 0.0532 | 0.0620 | +0.0087 |
| power strip | bookshelf | 39 | 1 | 39 | 0.0000 | 0.2229 | +0.2229 |
