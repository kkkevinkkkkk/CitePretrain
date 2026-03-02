from copy import deepcopy
import datetime
import pytz

SOT_TOKEN = "<|reserved_special_token_0|>"
EOT_TOKEN = "<|reserved_special_token_1|>"
SOD_TOKEN = "<|reserved_special_token_2|>"
EOD_TOKEN = "<|reserved_special_token_3|>"

triviaqa = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_citation": "Provide a clear and concise answer for the following question. Then provide citations for the information you used for your answer.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer. Make sure your output only includes the final answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Who was married to Spandau Ballet's Gary Kemp and later to Jude Law?",
            "answer": "Sadie Frost",
            "docs": [
                {
                    "title": "",
                    "text": 'From the moment Sadie Frost and Jude Law met on the set of 1992 Brit flick, Shopping, she felt it was her destiny to "spend the rest of my life" with him. Married to Spandau Ballet star Gary Kemp, Sadie, then 25, tried to "crush her unwelcome ideas" about Jude, knowing they were "jeopardising an idyllic home life."'
                }
            ]
        },
        {
            "question": "In which branch of the arts does Allegra Kent work?",
            "answer": "Ballet",
            "docs": [
                {
                    "title": "",
                    "text": "Allegra Kent (CBA '19), ballerina and muse of George Balanchine and Joseph Cornell, started studying ballet at 11 with Bronislava Nijinska and Carmelita Maracci. In 1952, Balanchine invited her to New York City Ballet, where she danced for the next 30 years."
                }
            ]
        },
        {
            "question": "Which animal is the national emblem of India?",
            "answer": "The Tiger",
            "docs": [
                {
                    "title": "",
                    "text": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India."
                }
            ]
        },
        {
            "question": "From which country did Angola achieve independence in 1975?",
            "answer": "Portugal",
            "docs": [
                {
                    "title": "",
                    "text": "Portugal granted Angola independence on November 11, 1975, at a time when multiple Angolan nationalist forces were fighting among themselves to establish control over the newly liberated state."
                }
            ]
        },
    ],
    # "demos_citation": [
    #     {
    #         "question": "Which ‘A’ is a liqueur originally distilled from wine and wormwood?",
    #         "answer": "Abstinthe\n[1] Artemisia absinthium_Part-1",
    #         "docs": [
    #             {
    #                 "title": "Artemisia absinthium_Part-1",
    #                 "text": "Artemisia absinthium, commonly known as wormwood, is an ingredient in the spirit absinthe and is used for flavoring in other spirits and wines, including bitters, vermouth, and pelinkovac. In the Middle Ages, it was used to spice mead, and in Morocco, it is used as tea."
    #             }]
    #     },
    #     {
    #         "question": "Who won the most men's US Open tennis titles in the 20th century?",
    #         "answer": "Bill Tilden\n[1] Prominent Figures in Early 20th Century Tennis",
    #         "docs": [
    #             {
    #                 "title": "Prominent Figures in Early 20th Century Tennis",
    #                 "text": "Bill Tilden’s career was marked by numerous accomplishments, particularly at the United States National Championships, now known as the US Open. He dominated the tournament during the 1920s, winning an unprecedented number of titles. Between 1920 and 1929, Tilden captured the men’s singles title seven times, making him the most successful male player in the history of the US Open during the 20th century. His victories came in 1920, 1921, 1922, 1923, 1924, 1925, and 1929, showcasing his sustained excellence and adaptability on the court."
    #             }]
    #     },
    #     {
    #         "question": "In Roman legend, which animals are said to have saved Rome from a surprise attack by raising an alarm?",
    #         "answer": "Barheaded geese\n[1] The Sacred Animals of Roman Legend",
    #         "docs": [
    #             {
    #                 "title": "The Sacred Animals of Roman Legend",
    #                 "text": "On that fateful night, as the Gauls began their stealthy advance, the sacred barheaded geese became agitated. Their honking, loud and persistent, cut through the silence of the night. The noise was so unusual and alarming that it woke the Roman soldiers, alerting them to the enemy’s presence. Thanks to the timely warning provided by the geese, the Romans were able to rally their defenses and repel the Gauls, saving the city from certain capture.",
    #             }
    #         ]
    #
    #     }
    # ],
    "demos_citation": [
        {
            "question": "Which ‘A’ is a liqueur originally distilled from wine and wormwood?",
            "answer": "Abstinthe\n[1] Artemisia absinthium",
            "docs": [
                {
                    "title": "Artemisia absinthium",
                    "text": "Artemisia absinthium, commonly known as wormwood, is an ingredient in the spirit absinthe and is used for flavoring in other spirits and wines, including bitters, vermouth, and pelinkovac. In the Middle Ages, it was used to spice mead, and in Morocco, it is used as tea."
                }]
        },
        {
            "question": "Who won the most men's US Open tennis titles in the 20th century?",
            "answer": "Bill Tilden\n[1] Prominent Figures in Early 20th Century Tennis",
            "docs": [
                {
                    "title": "Prominent Figures in Early 20th Century Tennis",
                    "text": "Bill Tilden’s career was marked by numerous accomplishments, particularly at the United States National Championships, now known as the US Open. He dominated the tournament during the 1920s, winning an unprecedented number of titles. Between 1920 and 1929, Tilden captured the men’s singles title seven times, making him the most successful male player in the history of the US Open during the 20th century. His victories came in 1920, 1921, 1922, 1923, 1924, 1925, and 1929, showcasing his sustained excellence and adaptability on the court."
                }]
        },
        {
            "question": "In Roman legend, which animals are said to have saved Rome from a surprise attack by raising an alarm?",
            "answer": "Barheaded geese\n[1] The Sacred Animals of Roman Legend",
            "docs": [
                {
                    "title": "The Sacred Animals of Roman Legend",
                    "text": "On that fateful night, as the Gauls began their stealthy advance, the sacred barheaded geese became agitated. Their honking, loud and persistent, cut through the silence of the night. The noise was so unusual and alarming that it woke the Roman soldiers, alerting them to the enemy’s presence. Thanks to the timely warning provided by the geese, the Romans were able to rally their defenses and repel the Gauls, saving the city from certain capture.",
                }
            ]

        }
    ],
    "demos_citation_st": [
        {
            "question": "Which ‘A’ is a liqueur originally distilled from wine and wormwood?",
            "answer": f"Abstinthe. {SOT_TOKEN}Artemisia absinthium{EOT_TOKEN}",
            "docs": [
                {
                    "title": "Artemisia absinthium_Part-1",
                    "text": "Artemisia absinthium, commonly known as wormwood, is an ingredient in the spirit absinthe and is used for flavoring in other spirits and wines, including bitters, vermouth, and pelinkovac. In the Middle Ages, it was used to spice mead, and in Morocco, it is used as tea."
                }]
        },
        {
            "question": "Who won the most men's US Open tennis titles in the 20th century?",
            "answer": f"Bill Tilden. {SOT_TOKEN}Prominent Figures in Early 20th Century Tennis{EOT_TOKEN}",
            "docs": [
                {
                    "title": "Prominent Figures in Early 20th Century Tennis",
                    "text": "Bill Tilden’s career was marked by numerous accomplishments, particularly at the United States National Championships, now known as the US Open. He dominated the tournament during the 1920s, winning an unprecedented number of titles. Between 1920 and 1929, Tilden captured the men’s singles title seven times, making him the most successful male player in the history of the US Open during the 20th century. His victories came in 1920, 1921, 1922, 1923, 1924, 1925, and 1929, showcasing his sustained excellence and adaptability on the court."
                }]
        },
        {
            "question": "In Roman legend, which animals are said to have saved Rome from a surprise attack by raising an alarm?",
            "answer": f"Barheaded geese. {SOT_TOKEN}The Sacred Animals of Roman Legend{EOT_TOKEN}",
            "docs": [
                {
                    "title": "The Sacred Animals of Roman Legend",
                    "text": "On that fateful night, as the Gauls began their stealthy advance, the sacred barheaded geese became agitated. Their honking, loud and persistent, cut through the silence of the night. The noise was so unusual and alarming that it woke the Roman soldiers, alerting them to the enemy’s presence. Thanks to the timely warning provided by the geese, the Romans were able to rally their defenses and repel the Gauls, saving the city from certain capture.",
                }
            ]

        }
    ]
    # "demos_citation_st": [
    #         {
    #             "question": "Which ‘A’ is a liqueur originally distilled from wine and wormwood?",
    #             "answer": f"Abstinthe\n[1] {SOT_TOKEN}Artemisia absinthium_Part-1{EOT_TOKEN}",
    #             "docs": [
    #                 {
    #                     "title": "Artemisia absinthium_Part-1",
    #                     "text": "Artemisia absinthium, commonly known as wormwood, is an ingredient in the spirit absinthe and is used for flavoring in other spirits and wines, including bitters, vermouth, and pelinkovac. In the Middle Ages, it was used to spice mead, and in Morocco, it is used as tea."
    #                 }]
    #         },
    #         {
    #             "question": "Who won the most men's US Open tennis titles in the 20th century?",
    #             "answer": f"Bill Tilden\n[1] {SOT_TOKEN}Prominent Figures in Early 20th Century Tennis{EOT_TOKEN}",
    #             "docs": [
    #                 {
    #                     "title": "Prominent Figures in Early 20th Century Tennis",
    #                     "text": "Bill Tilden’s career was marked by numerous accomplishments, particularly at the United States National Championships, now known as the US Open. He dominated the tournament during the 1920s, winning an unprecedented number of titles. Between 1920 and 1929, Tilden captured the men’s singles title seven times, making him the most successful male player in the history of the US Open during the 20th century. His victories came in 1920, 1921, 1922, 1923, 1924, 1925, and 1929, showcasing his sustained excellence and adaptability on the court."
    #                 }]
    #         },
    #         {
    #             "question": "In Roman legend, which animals are said to have saved Rome from a surprise attack by raising an alarm?",
    #             "answer": f"Barheaded geese\n[1] {SOT_TOKEN}The Sacred Animals of Roman Legend{EOT_TOKEN}",
    #             "docs": [
    #                 {
    #                     "title": "The Sacred Animals of Roman Legend",
    #                     "text": "On that fateful night, as the Gauls began their stealthy advance, the sacred barheaded geese became agitated. Their honking, loud and persistent, cut through the silence of the night. The noise was so unusual and alarming that it woke the Roman soldiers, alerting them to the enemy’s presence. Thanks to the timely warning provided by the geese, the Romans were able to rally their defenses and repel the Gauls, saving the city from certain capture.",
    #                 }
    #             ]
    #
    #         }
    #     ]
}

asqa = {
    "instruction": "You will be given an ambiguous factoid question that has different correct answers depending on the interpretation. Your answer should synthesize factual information from multiple sources into a long-form summary that resolves the ambiguity. Provide a clear and concise answer with an unbiased tone.",
    "instruction_citation": "You will be given an ambiguous factoid question that has different correct answers depending on the interpretation. Your answer should synthesize factual information from multiple sources into a long-form summary that resolves the ambiguity. Provide a clear and concise answer with an unbiased tone. Cite the sources you used to generate your answer right after each fact claim in the answer.",
    "instruction_citation2": "You will be given an ambiguous factoid question that has different correct answers depending on the interpretation. Your answer should synthesize factual information from multiple sources into a long-form summary that resolves the ambiguity. Provide a clear and concise answer with an unbiased tone. For every factual claim, cite the source immediately after the claim using the following format: <|reserved_special_token_0|> title <|reserved_special_token_1|>, where title is the title of the document that supports the claim.",
    "instruction_citation_rag": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
    "instruction_citation_rag_refusal": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. If the provided search results do not contain sufficient information to answer the question, say \"Provided search results do not contain sufficient information to answer the question.\"",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "doc_prompt": "Document [{ID}](Title: {T}): {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Which is the most rainy place on earth?",
            "answer": "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm, although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861.",
            "docs": [
                {
                    "title": "Cherrapunji",
                    "text": "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in"
                },
                {
                    "title": "Cherrapunji",
                    "text": "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall"
                },
                {
                    "title": "Mawsynram",
                    "text": "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′"
                },
                {
                    "title": "Earth rainfall climatology",
                    "text": "Pacific Northwest, and the Sierra Nevada range are the wetter portions of the nation, with average rainfall exceeding per year. The drier areas are the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern Utah, central Wyoming, eastern Oregon and Washington and the northeast of the Olympic Peninsula. The Big Bog on the island of Maui receives, on average, every year, making it the wettest location in the US, and all of Oceania. The annual average rainfall maxima across the continent lie across the northwest from northwest Brazil into northern Peru, Colombia, and Ecuador, then along the Atlantic coast of"
                },
                {
                    "title": "Going to Extremes",
                    "text": "in the world. Oymyakon in Siberia, where the average winter temperature is −47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive years without rain. Fog is the only local source of water. Mawsynram in India, where average annual rainfall is 14 meters, falling within a four-month period in the monsoon season. The rainfall is approximately equal to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the 'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over the year. In his second series, Middleton visited places without permanent towns, locations where \"survival\""
                }
            ]
        },
        {
            "question": "When did the us break away from england?",
            "answer": "The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress). The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire.",
            "docs": [
                {
                    "title": "United States withdrawal from Saudi Arabia",
                    "text": "United States withdrawal from Saudi Arabia Beginning during Operation Desert Shield in August 1990, while preparing for the Gulf War, the United States sent a large troop contingent to Saudi Arabia. After the war, remnant troops, primarily U.S. Air Force personnel, augmented by a smaller number of coordinating and training personnel from the U.S. Navy, U.S. Army and U.S. Marine Corps remained in Saudi Arabia under the aegis of Joint Task Force Southwest Asia (JTF-SWA), as part of Operation Southern Watch (OSW). The United Kingdom and France also maintained a small contingent of Royal Air Force and French Air Force"
                },
                {
                    "title": "Decolonization of the Americas",
                    "text": "and France has fully \"integrated\" most of its former colonies as fully constituent \"departments\" of France. The United States of America declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, the date when the Declaration of Independence was officially adopted by Congress), in so doing becoming the first independent, foreign-recognized nation in the Americas and the first European colonial entity to break from its mother country. Britain formally acknowledged American independence in 1783 after its defeat in the American Revolutionary War. Although initially occupying only the land east of the Mississippi"
                },
                {
                    "title": "American Revolution",
                    "text": "second British army at Yorktown in the fall of 1781, effectively ending the war. The Treaty of Paris was signed September 3, 1783, formally ending the conflict and confirming the new nation's complete separation from the British Empire. The United States took possession of nearly all the territory east of the Mississippi River and south of the Great Lakes, with the British retaining control of Canada and Spain taking Florida. Among the significant results of the revolution was the creation of the United States Constitution, establishing a relatively strong federal national government that included an executive, a national judiciary, and"
                },
                {
                    "title": "Decolonization",
                    "text": "accelerate decolonialization and bring an end to the colonial empires of its Western allies, most importantly during the 1956 Suez Crisis, but American military bases were established around the world and direct and indirect interventions continued in Korea, Indochina, Latin America (\"inter alia\", the 1965 occupation of the Dominican Republic), Africa, and the Middle East to oppose Communist invasions and insurgencies. Since the dissolution of the Soviet Union, the United States has been far less active in the Americas, but invaded Afghanistan and Iraq following the September 11 attacks in 2001, establishing army and air bases in Central Asia. Before"
                },
                {
                    "title": "Decolonization",
                    "text": "the responsibility of the United Kingdom (with a copy of the new constitution annexed), and finally, if approved, issuance of an Order of Council fixing the exact date of independence. After World War I, several former German and Ottoman territories in the Middle East, Africa, and the Pacific were governed by the UK as League of Nations mandates. Some were administered directly by the UK, and others by British dominions – Nauru and the Territory of New Guinea by Australia, South West Africa by the Union of South Africa, and Western Samoa by New Zealand. Egypt became independent in 1922,"
                }
            ]
        },
        {
            "question": "Who set the record for longest field goal?",
            "answer": "The record for the longest field goal in an NFL game was set by Matt Prater at 64 yards, but the record for the longest field goal at any level was 69 yards, kicked by collegiate kicker Ove Johansson in a 1976 Abilene Christian University football game against East Texas State University.",
            "docs": [
                {
                    "title": "Field goal",
                    "text": "toward its own end. The longest field goal kick in NFL history is 64 yards, a record set by Matt Prater on December 8, 2013. The previous record was 63, originally set by Tom Dempsey (1970) and then matched by Jason Elam (1998), Sebastian Janikowski (2011), David Akers (2012), and Graham Gano (2018). High school, college and most professional football leagues offer only a three-point field goal; however, some professional leagues have encouraged more rare kicks through \"four-point field goals\". NFL Europe encouraged long field goals of 50 yards or more by making those worth four points instead of three"
                },
                {
                    "title": "Field goal range",
                    "text": "35 and 40 yard lines (closer in a crosswind) often will go for the more risky fourth down conversion rather than risk either the touchback or the missed field goal. The longest field goal in recorded football history was 69 yards, set by collegiate kicker Ove Johansson, who was born in Sweden, in a 1976 Abilene Christian University football game against East Texas State University (now Texas A&M Commerce) at Shotwell Stadium in Abilene. The longest successful field goal in the NFL was 64 yards and was completed by Matt Prater in 2013. The NCAA record is 67 yards held"
                },
                {
                    "title": "Field goal",
                    "text": "both end zones) is only 66 yards. Scaccia, while playing indoor football, attempted a 64-yard kick that was inches short of success, hitting the crossbar. Longer field goals have been attempted at times; the longest attempt in the NFL, which was well short and was kicked into the wind, was 76 yards, attempted by Sebastian Janikowski of the Oakland Raiders, in a September 28, 2008 game against the San Diego Chargers. NFL Europe rewarded kickers that successfully kicked a field goal of longer than 50 yards with a bonus point, making such field goals worth 4 points instead of 3;"
                },
                {
                    "title": "Field goal",
                    "text": "this accomplishment is not the official record. All of the above kicks were successful with the use of a kicking tee, which was banned by the NCAA after the 1988 season. The longest known drop-kicked field goal in college football was a 62-yard kick from Pat O'Dea, an Australian kicker who played on the Wisconsin Badgers football team. O'Dea's kick took place in a blizzard against Northwestern on November 15, 1898. The longest field goal in U Sports football history is 59 yards, by Niko Difonte of Calgary Dinos, playing against the UBC Thunderbirds on November 11, 2017. The field"
                },
                {
                    "title": "Field goal range",
                    "text": "NFL and have been banned from NCAA since 1989) is 68 yards held by Fabrizio Scaccia, and the high school record 68 yards held by Dirk Borgognone; high school has wider goal posts and treats a field goal attempt that lands short in the field of play the same as a punt, making longer attempts much less risky. The indoor football record, with narrower and higher goal posts, is 63 yards (set by Aaron Mills), which is practically as long of a field goal as is possible in that variant of the sport, since the field in indoor football (including"
                }
            ]
        },
        {
            "question": "Who played galen in planet of the apes?",
            "answer": "In the 1968 film Planet of the Apes, Galen was played by Wright King. And in the tv series Planet of the Apes, Galen was played by Roddy McDowall.",
            "docs": [
                {
                    "title": "Planet of the Apes",
                    "text": "installment. Jacobs died on June 27, 1973, bringing an end to the APJAC Productions era of the \"Planet of the Apes\" franchise. Former Fox executive Stan Hough took over as producer for the television project, titled \"Planet of the Apes\". CBS picked up the series for its 1974 autumn lineup. Ron Harper and James Naughton played Alan Virdon and Peter Burke, two 20th-century American astronauts who pass through a time warp to a future where apes subjugate humans (unlike the original film, the humans can speak). Roddy McDowall returned to the franchise as Galen, a chimpanzee who joins the astronauts."
                },
                {
                    "title": "Planet of the Apes (1968 film)",
                    "text": "chimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright King). While unable to speak as his throat wound is healing, called \"Bright Eyes\" by Zira and placed with one of the captive primitive humans he later names \"Nova\", Taylor observes the enhanced society of talking apes and in a strict caste system: the gorillas being the military police, hunters and workers; the orangutans overseeing the affairs of government, science, and religion; and intellectual chimpanzees being mostly scientists. While their society is a theocracy similar to the beginnings of the human Industrial Era, the apes consider the primitive humans as"
                },
                {
                    "title": "Planet of the Apes (1968 film)",
                    "text": "Planet of the Apes (1968 film) Planet of the Apes is a 1968 American science fiction film directed by Franklin J. Schaffner. It stars Charlton Heston, Roddy McDowall, Kim Hunter, Maurice Evans, James Whitmore, James Daly and Linda Harrison. The screenplay by Michael Wilson and Rod Serling was loosely based on the 1963 French novel \"La Plan\u00e8te des Singes\" by Pierre Boulle. Jerry Goldsmith composed the groundbreaking avant-garde score. It was the first in a series of five films made between 1968 and 1973, all produced by Arthur P. Jacobs and released by 20th Century Fox. The film tells the"
                },
                {
                    "title": "Planet of the Apes",
                    "text": "Rupert Wyatt. To portray ape characters realistically, the production avoided practical effects in favor of performance capture acting, partnering with New Zealand visual effects company Weta Digital. Wyatt cast James Franco as Will Rodman, while veteran performance capture actor Andy Serkis signed on to star as Caesar. \"Rise\" debuted on August 5, 2011. Critics reviewed it positively, especially praising the visual effects and Serkis's performance. It was a major box office hit, taking in $482 million globally, more than five times its $93 million budget. Weta's special effects earned the film two Visual Effects Society Awards and an Oscar nomination"
                },
                {
                    "title": "Planet of the Apes",
                    "text": "film stars Mark Wahlberg as astronaut Leo Davidson, who accidentally travels through a wormhole to a distant planet where talking apes enslave humans. He leads a human revolt and upends ape civilization by discovering that the apes evolved from the normal earth primates who had accompanied his mission, and arrived years before. Helena Bonham Carter played chimpanzee Ari, while Tim Roth played the human-hating chimpanzee General Thade. The film received mixed reviews; most critics believed it failed to compare to the original. Much of the negative commentary focused on the confusing plot and twist ending, though many reviewers praised the"
                }
            ]
        }
    ],
    "demos_citation": [
        {
            "question": "Which is the most rainy place on earth?",
            "answer": "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012. <source> Mawsynram </source> However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm, although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861. <source> Cherrapunji </source>",
            "docs": [
                {
                    "title": "Cherrapunji",
                    "text": "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in"
                },
                {
                    "title": "Cherrapunji",
                    "text": "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall"
                },
                {
                    "title": "Mawsynram",
                    "text": "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′"
                },
                {
                    "title": "Earth rainfall climatology",
                    "text": "Pacific Northwest, and the Sierra Nevada range are the wetter portions of the nation, with average rainfall exceeding per year. The drier areas are the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern Utah, central Wyoming, eastern Oregon and Washington and the northeast of the Olympic Peninsula. The Big Bog on the island of Maui receives, on average, every year, making it the wettest location in the US, and all of Oceania. The annual average rainfall maxima across the continent lie across the northwest from northwest Brazil into northern Peru, Colombia, and Ecuador, then along the Atlantic coast of"
                },
                {
                    "title": "Going to Extremes",
                    "text": "in the world. Oymyakon in Siberia, where the average winter temperature is −47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive years without rain. Fog is the only local source of water. Mawsynram in India, where average annual rainfall is 14 meters, falling within a four-month period in the monsoon season. The rainfall is approximately equal to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the 'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over the year. In his second series, Middleton visited places without permanent towns, locations where \"survival\""
                }
            ]
        },
        {
            "question": "When did the us break away from england?",
            "answer": "The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress). <source> Decolonization of the Americas </source> The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire. <source> American Revolution </source>",
            "docs": [
                {
                    "title": "United States withdrawal from Saudi Arabia",
                    "text": "United States withdrawal from Saudi Arabia Beginning during Operation Desert Shield in August 1990, while preparing for the Gulf War, the United States sent a large troop contingent to Saudi Arabia. After the war, remnant troops, primarily U.S. Air Force personnel, augmented by a smaller number of coordinating and training personnel from the U.S. Navy, U.S. Army and U.S. Marine Corps remained in Saudi Arabia under the aegis of Joint Task Force Southwest Asia (JTF-SWA), as part of Operation Southern Watch (OSW). The United Kingdom and France also maintained a small contingent of Royal Air Force and French Air Force"
                },
                {
                    "title": "Decolonization of the Americas",
                    "text": "and France has fully \"integrated\" most of its former colonies as fully constituent \"departments\" of France. The United States of America declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, the date when the Declaration of Independence was officially adopted by Congress), in so doing becoming the first independent, foreign-recognized nation in the Americas and the first European colonial entity to break from its mother country. Britain formally acknowledged American independence in 1783 after its defeat in the American Revolutionary War. Although initially occupying only the land east of the Mississippi"
                },
                {
                    "title": "American Revolution",
                    "text": "second British army at Yorktown in the fall of 1781, effectively ending the war. The Treaty of Paris was signed September 3, 1783, formally ending the conflict and confirming the new nation's complete separation from the British Empire. The United States took possession of nearly all the territory east of the Mississippi River and south of the Great Lakes, with the British retaining control of Canada and Spain taking Florida. Among the significant results of the revolution was the creation of the United States Constitution, establishing a relatively strong federal national government that included an executive, a national judiciary, and"
                },
                {
                    "title": "Decolonization",
                    "text": "accelerate decolonialization and bring an end to the colonial empires of its Western allies, most importantly during the 1956 Suez Crisis, but American military bases were established around the world and direct and indirect interventions continued in Korea, Indochina, Latin America (\"inter alia\", the 1965 occupation of the Dominican Republic), Africa, and the Middle East to oppose Communist invasions and insurgencies. Since the dissolution of the Soviet Union, the United States has been far less active in the Americas, but invaded Afghanistan and Iraq following the September 11 attacks in 2001, establishing army and air bases in Central Asia. Before"
                },
                {
                    "title": "Decolonization",
                    "text": "the responsibility of the United Kingdom (with a copy of the new constitution annexed), and finally, if approved, issuance of an Order of Council fixing the exact date of independence. After World War I, several former German and Ottoman territories in the Middle East, Africa, and the Pacific were governed by the UK as League of Nations mandates. Some were administered directly by the UK, and others by British dominions – Nauru and the Territory of New Guinea by Australia, South West Africa by the Union of South Africa, and Western Samoa by New Zealand. Egypt became independent in 1922,"
                }
            ]
        },
        {
            "question": "Who set the record for longest field goal?",
            "answer": "The record for the longest field goal in an NFL game was set by Matt Prater at 64 yards, <source> Field goal </source> but the record for the longest field goal at any level was 69 yards, kicked by collegiate kicker Ove Johansson in a 1976 Abilene Christian University football game against East Texas State University. <source> Field goal </source>",
            "docs": [
                {
                    "title": "Field goal",
                    "text": "toward its own end. The longest field goal kick in NFL history is 64 yards, a record set by Matt Prater on December 8, 2013. The previous record was 63, originally set by Tom Dempsey (1970) and then matched by Jason Elam (1998), Sebastian Janikowski (2011), David Akers (2012), and Graham Gano (2018). High school, college and most professional football leagues offer only a three-point field goal; however, some professional leagues have encouraged more rare kicks through \"four-point field goals\". NFL Europe encouraged long field goals of 50 yards or more by making those worth four points instead of three"
                },
                {
                    "title": "Field goal range",
                    "text": "35 and 40 yard lines (closer in a crosswind) often will go for the more risky fourth down conversion rather than risk either the touchback or the missed field goal. The longest field goal in recorded football history was 69 yards, set by collegiate kicker Ove Johansson, who was born in Sweden, in a 1976 Abilene Christian University football game against East Texas State University (now Texas A&M Commerce) at Shotwell Stadium in Abilene. The longest successful field goal in the NFL was 64 yards and was completed by Matt Prater in 2013. The NCAA record is 67 yards held"
                },
                {
                    "title": "Field goal",
                    "text": "both end zones) is only 66 yards. Scaccia, while playing indoor football, attempted a 64-yard kick that was inches short of success, hitting the crossbar. Longer field goals have been attempted at times; the longest attempt in the NFL, which was well short and was kicked into the wind, was 76 yards, attempted by Sebastian Janikowski of the Oakland Raiders, in a September 28, 2008 game against the San Diego Chargers. NFL Europe rewarded kickers that successfully kicked a field goal of longer than 50 yards with a bonus point, making such field goals worth 4 points instead of 3;"
                },
                {
                    "title": "Field goal",
                    "text": "this accomplishment is not the official record. All of the above kicks were successful with the use of a kicking tee, which was banned by the NCAA after the 1988 season. The longest known drop-kicked field goal in college football was a 62-yard kick from Pat O'Dea, an Australian kicker who played on the Wisconsin Badgers football team. O'Dea's kick took place in a blizzard against Northwestern on November 15, 1898. The longest field goal in U Sports football history is 59 yards, by Niko Difonte of Calgary Dinos, playing against the UBC Thunderbirds on November 11, 2017. The field"
                },
                {
                    "title": "Field goal range",
                    "text": "NFL and have been banned from NCAA since 1989) is 68 yards held by Fabrizio Scaccia, and the high school record 68 yards held by Dirk Borgognone; high school has wider goal posts and treats a field goal attempt that lands short in the field of play the same as a punt, making longer attempts much less risky. The indoor football record, with narrower and higher goal posts, is 63 yards (set by Aaron Mills), which is practically as long of a field goal as is possible in that variant of the sport, since the field in indoor football (including"
                }
            ]
        },
        {
            "question": "Who played galen in planet of the apes?",
            "answer": "In the 1968 film Planet of the Apes, Galen was played by Wright King. <source> Planet of the Apes (1968 film) </source> And in the tv series Planet of the Apes, Galen was played by Roddy McDowall. <source> Planet of the Apes </source>",
            "docs": [
                {
                    "title": "Planet of the Apes",
                    "text": "installment. Jacobs died on June 27, 1973, bringing an end to the APJAC Productions era of the \"Planet of the Apes\" franchise. Former Fox executive Stan Hough took over as producer for the television project, titled \"Planet of the Apes\". CBS picked up the series for its 1974 autumn lineup. Ron Harper and James Naughton played Alan Virdon and Peter Burke, two 20th-century American astronauts who pass through a time warp to a future where apes subjugate humans (unlike the original film, the humans can speak). Roddy McDowall returned to the franchise as Galen, a chimpanzee who joins the astronauts."
                },
                {
                    "title": "Planet of the Apes (1968 film)",
                    "text": "chimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright King). While unable to speak as his throat wound is healing, called \"Bright Eyes\" by Zira and placed with one of the captive primitive humans he later names \"Nova\", Taylor observes the enhanced society of talking apes and in a strict caste system: the gorillas being the military police, hunters and workers; the orangutans overseeing the affairs of government, science, and religion; and intellectual chimpanzees being mostly scientists. While their society is a theocracy similar to the beginnings of the human Industrial Era, the apes consider the primitive humans as"
                },
                {
                    "title": "Planet of the Apes (1968 film)",
                    "text": "Planet of the Apes (1968 film) Planet of the Apes is a 1968 American science fiction film directed by Franklin J. Schaffner. It stars Charlton Heston, Roddy McDowall, Kim Hunter, Maurice Evans, James Whitmore, James Daly and Linda Harrison. The screenplay by Michael Wilson and Rod Serling was loosely based on the 1963 French novel \"La Plan\u00e8te des Singes\" by Pierre Boulle. Jerry Goldsmith composed the groundbreaking avant-garde score. It was the first in a series of five films made between 1968 and 1973, all produced by Arthur P. Jacobs and released by 20th Century Fox. The film tells the"
                },
                {
                    "title": "Planet of the Apes",
                    "text": "Rupert Wyatt. To portray ape characters realistically, the production avoided practical effects in favor of performance capture acting, partnering with New Zealand visual effects company Weta Digital. Wyatt cast James Franco as Will Rodman, while veteran performance capture actor Andy Serkis signed on to star as Caesar. \"Rise\" debuted on August 5, 2011. Critics reviewed it positively, especially praising the visual effects and Serkis's performance. It was a major box office hit, taking in $482 million globally, more than five times its $93 million budget. Weta's special effects earned the film two Visual Effects Society Awards and an Oscar nomination"
                },
                {
                    "title": "Planet of the Apes",
                    "text": "film stars Mark Wahlberg as astronaut Leo Davidson, who accidentally travels through a wormhole to a distant planet where talking apes enslave humans. He leads a human revolt and upends ape civilization by discovering that the apes evolved from the normal earth primates who had accompanied his mission, and arrived years before. Helena Bonham Carter played chimpanzee Ari, while Tim Roth played the human-hating chimpanzee General Thade. The film received mixed reviews; most critics believed it failed to compare to the original. Much of the negative commentary focused on the confusing plot and twist ending, though many reviewers praised the"
                }
            ]
        }

    ],
    "demos_citation_rag_refusal": [
    {
        "question": "Which is the most rainy place on earth?",
        "answer": "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861 [1].",
        "docs": [
            {
                "title": "Cherrapunji",
                "text": "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in"
            },
            {
                "title": "Cherrapunji",
                "text": "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall"
            },
            {
                "title": "Mawsynram",
                "text": "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′"
            },
            {
                "title": "Earth rainfall climatology",
                "text": "Pacific Northwest, and the Sierra Nevada range are the wetter portions of the nation, with average rainfall exceeding per year. The drier areas are the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern Utah, central Wyoming, eastern Oregon and Washington and the northeast of the Olympic Peninsula. The Big Bog on the island of Maui receives, on average, every year, making it the wettest location in the US, and all of Oceania. The annual average rainfall maxima across the continent lie across the northwest from northwest Brazil into northern Peru, Colombia, and Ecuador, then along the Atlantic coast of"
            },
            {
                "title": "Going to Extremes",
                "text": "in the world. Oymyakon in Siberia, where the average winter temperature is −47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive years without rain. Fog is the only local source of water. Mawsynram in India, where average annual rainfall is 14 meters, falling within a four-month period in the monsoon season. The rainfall is approximately equal to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the 'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over the year. In his second series, Middleton visited places without permanent towns, locations where \"survival\""
            }
        ]
    },
    {
        "question": "When did the us break away from england?",
        "answer": "The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress) [2]. The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire [3].",
        "docs": [
            {
                "title": "United States withdrawal from Saudi Arabia",
                "text": "United States withdrawal from Saudi Arabia Beginning during Operation Desert Shield in August 1990, while preparing for the Gulf War, the United States sent a large troop contingent to Saudi Arabia. After the war, remnant troops, primarily U.S. Air Force personnel, augmented by a smaller number of coordinating and training personnel from the U.S. Navy, U.S. Army and U.S. Marine Corps remained in Saudi Arabia under the aegis of Joint Task Force Southwest Asia (JTF-SWA), as part of Operation Southern Watch (OSW). The United Kingdom and France also maintained a small contingent of Royal Air Force and French Air Force"
            },
            {
                "title": "Decolonization of the Americas",
                "text": "and France has fully \"integrated\" most of its former colonies as fully constituent \"departments\" of France. The United States of America declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, the date when the Declaration of Independence was officially adopted by Congress), in so doing becoming the first independent, foreign-recognized nation in the Americas and the first European colonial entity to break from its mother country. Britain formally acknowledged American independence in 1783 after its defeat in the American Revolutionary War. Although initially occupying only the land east of the Mississippi"
            },
            {
                "title": "American Revolution",
                "text": "second British army at Yorktown in the fall of 1781, effectively ending the war. The Treaty of Paris was signed September 3, 1783, formally ending the conflict and confirming the new nation's complete separation from the British Empire. The United States took possession of nearly all the territory east of the Mississippi River and south of the Great Lakes, with the British retaining control of Canada and Spain taking Florida. Among the significant results of the revolution was the creation of the United States Constitution, establishing a relatively strong federal national government that included an executive, a national judiciary, and"
            },
            {
                "title": "Decolonization",
                "text": "accelerate decolonialization and bring an end to the colonial empires of its Western allies, most importantly during the 1956 Suez Crisis, but American military bases were established around the world and direct and indirect interventions continued in Korea, Indochina, Latin America (\"inter alia\", the 1965 occupation of the Dominican Republic), Africa, and the Middle East to oppose Communist invasions and insurgencies. Since the dissolution of the Soviet Union, the United States has been far less active in the Americas, but invaded Afghanistan and Iraq following the September 11 attacks in 2001, establishing army and air bases in Central Asia. Before"
            },
            {
                "title": "Decolonization",
                "text": "the responsibility of the United Kingdom (with a copy of the new constitution annexed), and finally, if approved, issuance of an Order of Council fixing the exact date of independence. After World War I, several former German and Ottoman territories in the Middle East, Africa, and the Pacific were governed by the UK as League of Nations mandates. Some were administered directly by the UK, and others by British dominions – Nauru and the Territory of New Guinea by Australia, South West Africa by the Union of South Africa, and Western Samoa by New Zealand. Egypt became independent in 1922,"
            }
        ]
    },
        {
            "question": "Who played galen in planet of the apes?",
            "answer": "Provided search results do not contain sufficient information to answer the question.",
            "docs": [
                {
                    "title": "Planet of the Apes Franchise Overview",
                    "text": "The Planet of the Apes franchise began with a 1963 novel and expanded into a series of films, television shows, comics, and merchandise. The central theme explores societal roles through a world where apes rule over humans. The original 1968 film is considered a classic, and the franchise has continued to gain new audiences with each adaptation."
                },
                {
                    "title": "Behind the Scenes of Planet of the Apes",
                    "text": "Production of the first Planet of the Apes film involved elaborate makeup techniques, innovative set design, and an international cast. Special features on the DVD release discuss the makeup transformation process but do not provide a full cast listing for specific characters."
                },
                {
                    "title": "Evolution of Science Fiction Films",
                    "text": "Science fiction cinema of the late 1960s and 1970s was defined by bold visual effects, social commentary, and memorable soundtracks. Planet of the Apes is cited for its cultural impact, but the discussion focuses on its genre influence, not individual actors."
                },
                {
                    "title": "Television Adaptations of Classic Sci-Fi",
                    "text": "Classic science fiction properties such as Planet of the Apes and Star Trek were adapted into television series in the 1970s. These adaptations often featured new storylines and different actors than the original films. While Planet of the Apes TV series is mentioned, no information about the character Galen or the actor is provided."
                },
                {
                    "title": "Memorable Characters in Film History",
                    "text": "Iconic film characters like Darth Vader, Ripley, and HAL 9000 are discussed for their influence on popular culture. While Planet of the Apes is mentioned as a source of memorable science fiction characters, there are no details about the character Galen or who portrayed him."
                }
            ]
        }
]
}
asqa["demos_citation_st"] = [deepcopy(demo) for demo in asqa["demos_citation"]]
asqa["demos_citation_rag"] = [deepcopy(demo) for demo in asqa["demos_citation"]]
asqa_citation_rag_answers = [
    "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861 [1].",
    "The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress) [2]. The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire [3].",
    "The record for the longest field goal in an NFL game was set by Matt Prater at 64 yards [1], but the record for the longest field goal at any level was 69 yards, kicked by collegiate kicker Ove Johansson in a 1976 Abilene Christian University football game against East Texas State University [2].",
    "In the 1968 film Planet of the Apes, Galen was played by Wright King [2]. And in the tv series Planet of the Apes, Galen was played by Roddy McDowall [1].",
]
for demo in asqa["demos_citation_st"]:
    demo["answer"] = demo["answer"].replace("<source>", SOT_TOKEN).replace("</source>", EOT_TOKEN)
for i, demo in enumerate(asqa["demos_citation_rag"]):
    demo["answer"] = asqa_citation_rag_answers[i]


eli5 = {
    "instruction": "Instruction: Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone. ",
    "instruction_citation": "Instruction: Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone. Use citations to support each of your factual claims.",
    "instruction_citation2": "Instruction: Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone. For every factual claim, cite the source immediately after the claim using the following format: <|reserved_special_token_0|> title <|reserved_special_token_1|>, where title is the title of the document that supports the claim.",
    "instruction_citation_rag": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
    "instruction_citation_rag_refusal": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. If the provided search results do not contain sufficient information to answer the question, say \"Provided search results do not contain sufficient information to answer the question.\"",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "doc_prompt": "Document [{ID}](Title: {T}): {P}\n\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Why did New York City try to ban food donations to the poor?",
            "answer": "New York City, under Mayor Michael Bloomberg's administration, banned citizens from donating food directly to homeless shelters because the city could not assess the salt, fat, and fiber content. Bloomberg's administration was heavily criticized for losing their common sense by becoming too focused on what people eat.",
            "docs": [
                {
                    "title": "The Future Of America",
                    "text": "believe that they are \u201chelping\u201d the homeless by passing such laws. In New York City, Mayor Bloomberg has banned citizens from donating food directly to homeless shelters and he is actually convinced that it was the right thing to do for the homeless\u2026 Mayor Michael Bloomberg\u2019s food police have struck again! Outlawed are food donations to homeless shelters because the city can\u2019t assess their salt, fat and fiber content, reports CBS 2\u2019s Marcia Kramer. Glenn Richter arrived at a West Side synagogue on Monday to collect surplus bagels \u2014 fresh nutritious bagels \u2014 to donate to the poor."
                },
                {
                    "title": "mayor bloomberg",
                    "text": "Amuck: Bloomberg Bans Food Donations in New York City Food Might Be Salty or Too High in Calories, City Explains Washington, D.C. \u2013 New York Mayor Michael Bloomberg\u2019s administration is now banning all food being offered to the city\u2019s homeless shelters. New York City\u2019s bureaucrats have become so singularly focused on what people eat, says the National Center for Public Policy Research, that they\u2019ve lost their common sense. \u201cSo much for serving the homeless: The Bloomberg administration is now taking the term \u2018food police\u2019 to new depths, blocking food donations to all government-run facilities that serve the"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND Front Page Health U.S. New York City bans food donations Inability to control 'nutritional content' cited as reason New York City homeless shelters have Mayor Michael Bloomberg to thank for a halt in food donations, for which hungry families are waiting, according to one public policy advocate. \"The Bloomberg administration is now taking the term 'food police' to new depths, blocking food donations to all government-run facilities that serve the city's homeless,\" says Jeff Stier, a National Center for Public Policy Research senior fellow. Currently, no food can be given to government-run, New York City facilities, despite hungry crowds perfectly"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND Services didn't return WND calls. Stier told WND that he specifically was told by Diamond that the policy was tied to the nutritional guidelines set by the mayor. \"They can say that this ban on donations is a long-standing policy, but they can\u2019t document it,\" Stier told WND. \"I've also been told that there are numerous food shelves that have been accepting food donations, not just one.\" Stier is a member of a New York Synagogue that has donated food for over a decade. He is outraged that the DHS' response to his demand to know why the practice can"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND ban on donated food. In fact, it thrives because of food donations. New York City Rescue Mission has been providing food, clothing, shelter and spiritual hope for needy New Yorkers since 1872. \"We feed over 500 people a day, all through donations,\" said James Varnhagen, NYCRM director. \"Boxed food, canned food, prepared food, we take any food,\" he told WND. \"We couldn't survive without donations,\" he said."
                }
            ]
        },
        {
            "question": "What's the difference between Shia vs. Sunni Islam?",
            "answer": "The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings, while the Shia branch follows Prophet Muhammad's son-in-law Ali. Nowadays, Sunni and Shia are the major branches of Islam.",
            "docs": [
                {
                    "title": "The Sunni vs Shia Divide - Explained - Globaloi",
                    "text": "centuries-long strained relationship between Sunnis and Shias. As a scholar of Islam and a public educator, I often field questions about Sunnis, Shias and the sects of Islam. What exactly is the Shia-Sunni divide? And what is its history? History of divide Both Sunnis and Shias \u2013 drawing their faith and practice from the Qur\u2019an and the life of the Prophet Muhammad \u2013 agree on most of the fundamentals of Islam. The differences are related more to historical events, ideological heritage and issues of leadership. The first and central difference emerged after the death of Prophet Muhammad in A.D. 632."
                },
                {
                    "title": "What\u2019s the difference between Sunni and Shia Islam? \u2013 Macrosnaps",
                    "text": "What\u2019s the difference between Sunni and Shia Islam? Sunni and Shia identities (the 2 main branches of Islam) first formed around a dispute over leadership succession after the death of the Prophet Muhammad in 632 A.D. Sunni is the larger branch (estimated 85-90% of total world Muslim population) and it's adherents are referred to as \"people of the tradition of Muhammad\", while Shia are \"followers\" of Muhammad's son-in-law and cousin Ali. Sunnis rely heavily on the practice of the Prophet Muhammad and his teachings, the Shia view their ayatollahs as reflections of God on earth. What challenges does the anti-IS"
                },
                {
                    "title": "Difference between Sunni and Shia Muslims | Sunni vs Shia Muslims",
                    "text": "of Muhammad, the last prophet of God. A follower of Islam is known as a Muslim. Many Muslims believe that their sole purpose is to worship and serve God, for which they have established five pillars of Islam that guides a Muslim on almost every aspect of life and society. Due to differences, Muslims have been divided into two primary sects: The Sunnis and the Shias. These two sects have many similarities and both consider themselves are Muslims, following the will of God. However, they are also different from each other in certain aspects. Both the Sunnis and the Shias,"
                },
                {
                    "title": "What is the difference between Shia and Sunni Islam? - Islam Stack Exchange",
                    "text": "What is the difference between Shia and Sunni Islam? - Islam Stack Exchange between Mutah marriage and Misyar marriage? What theological and historical factors distinguish Ibadi Islam from either Shia or Sunni schools? What are the principle/fundamental differences between Sunni and Shia? Nikah between a Sunni girl and Shia boy What is the difference between \u201cMubtalat-of-Wudu\u201d of Shia and Sunni? How can the Hadith be reliable when Sunnis and Shia follow different points of reference? Rejection of Mutawatir Hadith in Sunni Islam and Shia Islam"
                },
                {
                    "title": "What is the difference between Sunni and Shia Islam? | Patrick Syder Travel",
                    "text": "What is the difference between Sunni and Shia Islam? | Patrick Syder Travel What is the difference between Sunni and Shia Islam? This Channel 4 link answers some of the key questions about the difference between Sunni and Shia Islam and alarmingly, the politics on what is happening and why, in Syria\u2026\u2026. http://www.channel4.com/news/sunni-shia-islam-muslim-syria-middle-east-key-questions \u2190 Ethiopia Appeal \u2013 Help sponsor a nurse to train and to help others G\u00f6bekli Tepe, Turkey: a new wonder of the ancient world by Jeremy Seal (Telegraph Travel Section 23/04/2013) \u2192"
                }
            ]
        },
        {
            "question": "What causes Bi-polar disorder?",
            "answer": "Bipolar disorder is an emotional disorder that causes extreme mood swings between excitement and depression. The spectrum of mood swing may span from days to months. We are still not certain of the exact factors that cause such disorder, but genetics is considered a major factor.",
            "docs": [
                {
                    "title": "Bi-polar disorder | definition of Bi-polar disorder by Medical dictionary",
                    "text": "bi-polar disorder | definition of bi-polar disorder by medical dictionary https://medical-dictionary.thefreedictionary.com/bi-polar+disorder (redirected from bi-polar disorder) related to bi-polar disorder: depression bipolar disorder, formerly known as manic depression, is a mood disorder that causes radical emotional changes and mood swings, from manic, restless highs to depressive, listless lows. most bipolar individuals experience alternating episodes of mania and depression. bipolar disorder is characterized by alternating manic episodes in which the individual feels abnormally euphoric, optimistic, and energetic and depressive periods in which the individual feels sad, hopeless, guilty, and sometimes suicidal. manic or depressive periods may last for days, weeks, or months"
                },
                {
                    "title": "Mania and Bi-Polar",
                    "text": "can go from depressed to \u201csuper happy\u201d all in one day, or even in a few days, does not have a bi-polar disorder Bi-polar looks different depending on the severity of the symptoms. Most bi-polar diagnoses that are made are for bi-polar 2, with bi-polar 1 being much more rare. Bi-polar 1 is so severe that the individual will have periods of such agitation, or such reckless and seemingly foolish behavior that they put themselves or those around them in danger. It is not completely clear what causes bi-polar, but genetics seem to have a large role. The biggest factor"
                },
                {
                    "title": "Bi-Polar disorder",
                    "text": "Bi-Polar disorder Bi-polar is generally a cyclic disease where individuals display depressive and elevated episodes at regular intervals. It is a disorder resulting from the imbalance of the chemicals in the brain that causes a lot of fluctuations of mood. It is a fact that we all experience happy and sad moods, but people with bi-polar disorder experience the changes in mood at an increased level. The cause of this disorder is not known completely. However, it is estimated that there are different factors responsible for it. It is often connected to a genetic component. People suffering from the Bi-polar disorder are"
                },
                {
                    "title": "For Individuals \u2014 Adam Schwartz",
                    "text": "For Individuals \u2014 Adam Schwartz The information is extensive and covers a huge range of topics. Some of the topics include the different types of bi-polar, what it feels like, signs and symptoms, treatments and more. Black Dog Institute bi-polar causes resource specifically covers the variety of areas that could potentially be a cause of bi-polar disorder. Including genetics, environmental factors, pregnancy, and more. Black Dog Institute bi-polar treatments resource specifically covers multiple potential treatments options for bi-polar. Including management, types of psychological treatment, lifestyle changes, and more. Black Dog Institute bi-polar self-test resource is a short self-test for people who may be concerned if"
                },
                {
                    "title": "Depression Bi-polar Disorder Symptoms 2019 | Win Over Depression",
                    "text": "Depression Bi-polar Disorder Symptoms 2019 | Win Over Depression signs and symptoms of bipolar disorder. Learn more about the common symptoms of bipolar depression that some patients may experience. Home \u00bb Trending Health News \u00bb 10 Warning Signs of Bipolar Disorder: Depression. One of the most serious symptoms of bipolar disorder is. Bi Polar Depression. SEVERE SWINGS What is bipolar disorder, is it the same as manic depression, what are the symptoms and is there a cure? Bipolar disorder, or manic depression, causes symptoms of mania and depression. Read about bipolar disorder treatment, medications, and causes of this. Learn more about the different types of bipolar disorder. Find out"
                }
            ]
        },
        {
            "question": "How do student loans affect getting a mortgage?",
            "answer": "When applying for a mortgage, student loans can affect the debt to income ratio, which is a key factor in determining the amount that an individual can afford to pay for the mortgage. While student loan repayments do not appear in an individual's credit history and do not affect credit scores, lenders do consider the amount of an individual's student loan repayments when assessing their mortgage application. Some 83% of non-homeowners say student loan debt is preventing them from buying a home, according to the National Association of Realtors. It is important to note that student loans do not prevent an individual from getting a mortgage.",
            "docs": [
                {
                    "title": "Student Loans \u2013 How do they work? | The Financial Review",
                    "text": "typical debt. Student loan repayments do not appear in an individual\u2019s credit history, therefore there are no implications whatsoever. This also extends to applications for credit cards \u2013 student \u2018loans\u2019 are not acknowledged. One noteworthy aspect that is affected by student loans however, is mortgage applications. Nevertheless, it does not prevent an individual from getting a mortgage. For example, lenders will consider the amount of an individual\u2019s student loan repayments in order to assess the debt to income ratio and therefore establish the amount that the individual can afford to pay for the mortgage. Just as they do with other"
                },
                {
                    "title": "How Does Student Loan Debt Affect Buying a Home? | Experian",
                    "text": "Rates & Affordability How Student Loans Affect Getting a Mortgage Student Loan Impact on Credit Scores Other Factors for Getting Approved for a Mortgage If you're a recent college grad and hope to become a homeowner in the near future, you should know that student loan debt could affect buying a home by making it more difficult to get a mortgage. Some 83% of non-homeowners say student loan debt is preventing them from buying a home, according to the National Association of Realtors (NAR). But while student loan payments can make it harder to save for a down payment on"
                },
                {
                    "title": "Studentloanify - How your student loans affect your home mortgage prospects",
                    "text": "Though it may not seem fair, your student loan situation impacts your home mortgage outlook. Many people carry student loan debt, but it\u2019s the amount of the loan and how you handle your student loan repayment plan that will influence your ability to get a home mortgage as well as what your interest rate will be. Here are some specific factors about your student loan that will affect your home mortgage prospects. On your mortgage loan application, you will have to report how much your monthly student loan payment is. This amount will be deducted from your monthly gross income"
                },
                {
                    "title": "How do student loans affect your credit score? | Student Loan Planner",
                    "text": "How do student loans affect your credit score? | Student Loan Planner Your credit score is the three-digit number that dictates a lot in your adult life. Whether you\u2019re applying for a mortgage or looking to get an auto loan, this seemingly arbitrary number determines whether you get approved for a loan and also affects your interest rate. If you\u2019re a student loan borrower you may wonder, \u201cDo student loans affect credit score?\u201d You might be especially curious if you\u2019re in the process of applying for a mortgage. Here\u2019s how student loans affect your credit score and what to know for big life events, like getting a mortgage. Do student loans affect"
                },
                {
                    "title": "Does Student Loan Debt Affect Getting A Mortgage?",
                    "text": "Does Student Loan Debt Affect Getting A Mortgage? Home \u00bb Does Student Loan Debt Affect Getting A Mortgage? Last year, I helped answer a reader\u2019s question about applying for a mortgage while on Income Based Repayment. However, over the last several months, I\u2019ve been getting bombarded with questions about how student loan debt impacts your ability to get a mortgage. Maybe it\u2019s because the housing market is improving, or maybe it\u2019s because people are finally taking their student loan debt seriously. Anyway, I wanted to share a few reader questions and then look at whether student loan debt affects getting a mortgage. Here are the reader questions I\u2019ve"
                }
            ]
        }
    ],

"demos_citation_st": [
        {
            "question": "What's the stereotypical Japanese riff from?",
            "answer": f"The stereotypical Japanese riff is commonly known as the 'Oriental Riff.'<|reserved_special_token_0|>One-Hit Wonder Status Rejected: The Vapors' Underrated 80s Punk Sound<|reserved_special_token_1|> The 'Oriental Riff' is often associated with Asian cultures in Western media despite its vague and generalized representation.<|reserved_special_token_0|>Uncovering the Origins of the Oriental Riff: A Musical Stereotype<|reserved_special_token_1|> The origins of the 'Oriental Riff' are not entirely clear.<|reserved_special_token_0|>Uncovering the Origins of the Oriental Riff: A Musical Stereotype<|reserved_special_token_1|> The first known recorded use of the 'Oriental Riff' dates back to the stage show 'The Grand Chinese Spectacle of Aladdin or The Wonderful Lamp' in 1847.<|reserved_special_token_0|>Unpacking the Orientalist Riff: A Persistent Example of White Cultural Appropriation and Racism<|reserved_special_token_1|> The story of Aladdin is set in western China, despite Aladdin being a character from Arabian Nights.<|reserved_special_token_0|>Debunking the Myths: The Surprising Origins of Aladdin and the Arabian Nights<|reserved_special_token_1|> Over time, the 'Oriental Riff' has been used broadly in popular culture to evoke an 'Asian' theme.<|reserved_special_token_0|>Uncovering the Origins of the Oriental Riff: A Musical Stereotype<|reserved_special_token_1|> The 'Oriental Riff' is a stereotype rather than an authentic representation of any specific Asian music tradition.<|reserved_special_token_0|>Debunking the Orientalist Riff: Cultural Imperialism in the UK<|reserved_special_token_1|>",
            "docs": []
        },
        {
            "question": "When did we start using small denominations of time (sec, min, hour), and why is it so uniform between countries?",
            "answer": f"he use of small denominations of time such as seconds, minutes, and hours became more standardized with the development of clocks.<|reserved_special_token_0|>Understanding the Evolution of Time Measurement: From Sundials to Atomic Clocks<|reserved_special_token_1|> During the 16th and 17th centuries, mechanical clocks with hour hands were becoming more common.<|reserved_special_token_0|>Understanding Military Time: A Comprehensive Guide to Converting and Using 24-Hour Clocks<|reserved_special_token_1|> By the late 17th century, clocks with both hour and minute hands were widely in use.<|reserved_special_token_0|>Anatomy of a Clock Face: History, Design, and Functionality<|reserved_special_token_1|> By the 18th century, clocks featuring second hands became popular.<|reserved_special_token_0|>Evolution of Timekeeping: A History from Sundials to Smartwatches at WritingsCentre<|reserved_special_token_1|> The uniformity of measuring time in seconds, minutes, and hours across different countries developed due to the needs of industrialization and the expansion of railroads in the 19th century.<|reserved_special_token_0|>A Brief History of Clocks: How Humanity's Quest for Timekeeping Shaped Science and Technology<|reserved_special_token_1|> As railroads spread and schedules became crucial for coordinating transportation and communication, it was essential to have a consistent timekeeping system.<|reserved_special_token_0|>How Train Orders and Signaling Systems Ruled the Railroad: A Look at Early Safety Measures on North American Railroads<|reserved_special_token_1|> The Industrial Revolution emphasized the need for precise timekeeping to regulate work shifts, production schedules, and business operations, contributing to the widespread adoption of standardized time measurements globally.<|reserved_special_token_0|>The History of Daylight Savings Time: A Global Perspective from Ancestral Findings<|reserved_special_token_1|>",
            "docs": []
        },
        {
            "question": "Why was salt so valuable in the past?",
            "answer": f"Salt was extremely valuable in the past due to its role in preserving food and maintaining human health.<|reserved_special_token_0|>Most Valuable Resources in West Africa: Salt, Oil, Gold, and More<|reserved_special_token_1|> Ancient methods of salt extraction were labor-intensive and depended on specific geographical conditions.<|reserved_special_token_0|>The Art of Salt: Unlocking the Flavors and Controversies of Himalayan, Table, Kosher, and Infused Salts<|reserved_special_token_1|> Salt was typically obtained from evaporating seawater, which required suitable coastal areas with clay soil.<|reserved_special_token_0|>Navigating Medieval Markets: A Guide to Goods, Guilds, and Social Hierarchy in the Realm of Myth<|reserved_special_token_1|> Successful evaporation of seawater for salt required a long dry season without dilution from rain.<|reserved_special_token_0|>Global Mining Industry Trends and Developments in 1994: Geopolitics, Exploration, and Environmental Concerns<|reserved_special_token_1|> Consistent wind was necessary to aid the drying process in salt production.<|reserved_special_token_0|>San Francisco Bay Conservation and Development Plan for Sustainable Use and Protection<|reserved_special_token_1|> Salt production was geographically limited, which increased its value.<|reserved_special_token_0|>Robert Redding: A Renaissance Man - Trucking Stories, Financial Insights, and Paranormal Explorations<|reserved_special_token_1|> Salt was crucial for preserving food before refrigeration.<|reserved_special_token_0|>Most Valuable Resources in West Africa: Salt, Oil, Gold, and More<|reserved_special_token_1|> Salt was essential for seasoning, heightening its demand.<|reserved_special_token_0|>Most Valuable Resources in West Africa: Salt, Oil, Gold, and More<|reserved_special_token_1|> Salt was often traded like a precious commodity.<|reserved_special_token_0|>Salt and Gold: The Valuable Trade that Shaped West Africa's Kingdoms - Xautoman<|reserved_special_token_1|> Salt was used as a form of currency in some cultures.<|reserved_special_token_0|>Salt and Gold: The Valuable Trade that Shaped West Africa's Kingdoms - Xautoman<|reserved_special_token_1|>",
            "docs": [
            ]

        }
    ],
    "demos_citation_rag": [
        {
            "question": "Why did New York City try to ban food donations to the poor?",
            "answer": "New York City, under Mayor Michael Bloomberg's administration, banned citizens from donating food directly to homeless shelters because the city could not assess the salt, fat, and fiber content [1][2][3]. Bloomberg's administration was heavily criticized for losing their common sense by becoming too focused on what people eat [2].",
            "docs": [
                {
                    "title": "The Future Of America",
                    "text": "believe that they are \u201chelping\u201d the homeless by passing such laws. In New York City, Mayor Bloomberg has banned citizens from donating food directly to homeless shelters and he is actually convinced that it was the right thing to do for the homeless\u2026 Mayor Michael Bloomberg\u2019s food police have struck again! Outlawed are food donations to homeless shelters because the city can\u2019t assess their salt, fat and fiber content, reports CBS 2\u2019s Marcia Kramer. Glenn Richter arrived at a West Side synagogue on Monday to collect surplus bagels \u2014 fresh nutritious bagels \u2014 to donate to the poor."
                },
                {
                    "title": "mayor bloomberg",
                    "text": "Amuck: Bloomberg Bans Food Donations in New York City Food Might Be Salty or Too High in Calories, City Explains Washington, D.C. \u2013 New York Mayor Michael Bloomberg\u2019s administration is now banning all food being offered to the city\u2019s homeless shelters. New York City\u2019s bureaucrats have become so singularly focused on what people eat, says the National Center for Public Policy Research, that they\u2019ve lost their common sense. \u201cSo much for serving the homeless: The Bloomberg administration is now taking the term \u2018food police\u2019 to new depths, blocking food donations to all government-run facilities that serve the"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND Front Page Health U.S. New York City bans food donations Inability to control 'nutritional content' cited as reason New York City homeless shelters have Mayor Michael Bloomberg to thank for a halt in food donations, for which hungry families are waiting, according to one public policy advocate. \"The Bloomberg administration is now taking the term 'food police' to new depths, blocking food donations to all government-run facilities that serve the city's homeless,\" says Jeff Stier, a National Center for Public Policy Research senior fellow. Currently, no food can be given to government-run, New York City facilities, despite hungry crowds perfectly"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND Services didn't return WND calls. Stier told WND that he specifically was told by Diamond that the policy was tied to the nutritional guidelines set by the mayor. \"They can say that this ban on donations is a long-standing policy, but they can\u2019t document it,\" Stier told WND. \"I've also been told that there are numerous food shelves that have been accepting food donations, not just one.\" Stier is a member of a New York Synagogue that has donated food for over a decade. He is outraged that the DHS' response to his demand to know why the practice can"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND ban on donated food. In fact, it thrives because of food donations. New York City Rescue Mission has been providing food, clothing, shelter and spiritual hope for needy New Yorkers since 1872. \"We feed over 500 people a day, all through donations,\" said James Varnhagen, NYCRM director. \"Boxed food, canned food, prepared food, we take any food,\" he told WND. \"We couldn't survive without donations,\" he said."
                }
            ]
        },
        {
            "question": "What's the difference between Shia vs. Sunni Islam?",
            "answer": "The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership [1]. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. [1][2]. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings, while the Shia branch follows Prophet Muhammad's son-in-law Ali [2]. Nowadays, Sunni and Shia are the major branches of Islam [3].",
            "docs": [
                {
                    "title": "The Sunni vs Shia Divide - Explained - Globaloi",
                    "text": "centuries-long strained relationship between Sunnis and Shias. As a scholar of Islam and a public educator, I often field questions about Sunnis, Shias and the sects of Islam. What exactly is the Shia-Sunni divide? And what is its history? History of divide Both Sunnis and Shias \u2013 drawing their faith and practice from the Qur\u2019an and the life of the Prophet Muhammad \u2013 agree on most of the fundamentals of Islam. The differences are related more to historical events, ideological heritage and issues of leadership. The first and central difference emerged after the death of Prophet Muhammad in A.D. 632."
                },
                {
                    "title": "What\u2019s the difference between Sunni and Shia Islam? \u2013 Macrosnaps",
                    "text": "What\u2019s the difference between Sunni and Shia Islam? Sunni and Shia identities (the 2 main branches of Islam) first formed around a dispute over leadership succession after the death of the Prophet Muhammad in 632 A.D. Sunni is the larger branch (estimated 85-90% of total world Muslim population) and it's adherents are referred to as \"people of the tradition of Muhammad\", while Shia are \"followers\" of Muhammad's son-in-law and cousin Ali. Sunnis rely heavily on the practice of the Prophet Muhammad and his teachings, the Shia view their ayatollahs as reflections of God on earth. What challenges does the anti-IS"
                },
                {
                    "title": "Difference between Sunni and Shia Muslims | Sunni vs Shia Muslims",
                    "text": "of Muhammad, the last prophet of God. A follower of Islam is known as a Muslim. Many Muslims believe that their sole purpose is to worship and serve God, for which they have established five pillars of Islam that guides a Muslim on almost every aspect of life and society. Due to differences, Muslims have been divided into two primary sects: The Sunnis and the Shias. These two sects have many similarities and both consider themselves are Muslims, following the will of God. However, they are also different from each other in certain aspects. Both the Sunnis and the Shias,"
                },
                {
                    "title": "What is the difference between Shia and Sunni Islam? - Islam Stack Exchange",
                    "text": "What is the difference between Shia and Sunni Islam? - Islam Stack Exchange between Mutah marriage and Misyar marriage? What theological and historical factors distinguish Ibadi Islam from either Shia or Sunni schools? What are the principle/fundamental differences between Sunni and Shia? Nikah between a Sunni girl and Shia boy What is the difference between \u201cMubtalat-of-Wudu\u201d of Shia and Sunni? How can the Hadith be reliable when Sunnis and Shia follow different points of reference? Rejection of Mutawatir Hadith in Sunni Islam and Shia Islam"
                },
                {
                    "title": "What is the difference between Sunni and Shia Islam? | Patrick Syder Travel",
                    "text": "What is the difference between Sunni and Shia Islam? | Patrick Syder Travel What is the difference between Sunni and Shia Islam? This Channel 4 link answers some of the key questions about the difference between Sunni and Shia Islam and alarmingly, the politics on what is happening and why, in Syria\u2026\u2026. http://www.channel4.com/news/sunni-shia-islam-muslim-syria-middle-east-key-questions \u2190 Ethiopia Appeal \u2013 Help sponsor a nurse to train and to help others G\u00f6bekli Tepe, Turkey: a new wonder of the ancient world by Jeremy Seal (Telegraph Travel Section 23/04/2013) \u2192"
                }
            ]
        },
        {
            "question": "What causes Bi-polar disorder?",
            "answer": "Bipolar disorder is an emotional disorder that causes extreme mood swings between excitement and depression [1][3]. The spectrum of mood swing may span from days to months [1][2]. We are still not certain of the exact factors that cause such disorder, but genetics is considered a major factor [2][3].",
            "docs": [
                {
                    "title": "Bi-polar disorder | definition of Bi-polar disorder by Medical dictionary",
                    "text": "bi-polar disorder | definition of bi-polar disorder by medical dictionary https://medical-dictionary.thefreedictionary.com/bi-polar+disorder (redirected from bi-polar disorder) related to bi-polar disorder: depression bipolar disorder, formerly known as manic depression, is a mood disorder that causes radical emotional changes and mood swings, from manic, restless highs to depressive, listless lows. most bipolar individuals experience alternating episodes of mania and depression. bipolar disorder is characterized by alternating manic episodes in which the individual feels abnormally euphoric, optimistic, and energetic and depressive periods in which the individual feels sad, hopeless, guilty, and sometimes suicidal. manic or depressive periods may last for days, weeks, or months"
                },
                {
                    "title": "Mania and Bi-Polar",
                    "text": "can go from depressed to \u201csuper happy\u201d all in one day, or even in a few days, does not have a bi-polar disorder Bi-polar looks different depending on the severity of the symptoms. Most bi-polar diagnoses that are made are for bi-polar 2, with bi-polar 1 being much more rare. Bi-polar 1 is so severe that the individual will have periods of such agitation, or such reckless and seemingly foolish behavior that they put themselves or those around them in danger. It is not completely clear what causes bi-polar, but genetics seem to have a large role. The biggest factor"
                },
                {
                    "title": "Bi-Polar disorder",
                    "text": "Bi-Polar disorder Bi-polar is generally a cyclic disease where individuals display depressive and elevated episodes at regular intervals. It is a disorder resulting from the imbalance of the chemicals in the brain that causes a lot of fluctuations of mood. It is a fact that we all experience happy and sad moods, but people with bi-polar disorder experience the changes in mood at an increased level. The cause of this disorder is not known completely. However, it is estimated that there are different factors responsible for it. It is often connected to a genetic component. People suffering from the Bi-polar disorder are"
                },
                {
                    "title": "For Individuals \u2014 Adam Schwartz",
                    "text": "For Individuals \u2014 Adam Schwartz The information is extensive and covers a huge range of topics. Some of the topics include the different types of bi-polar, what it feels like, signs and symptoms, treatments and more. Black Dog Institute bi-polar causes resource specifically covers the variety of areas that could potentially be a cause of bi-polar disorder. Including genetics, environmental factors, pregnancy, and more. Black Dog Institute bi-polar treatments resource specifically covers multiple potential treatments options for bi-polar. Including management, types of psychological treatment, lifestyle changes, and more. Black Dog Institute bi-polar self-test resource is a short self-test for people who may be concerned if"
                },
                {
                    "title": "Depression Bi-polar Disorder Symptoms 2019 | Win Over Depression",
                    "text": "Depression Bi-polar Disorder Symptoms 2019 | Win Over Depression signs and symptoms of bipolar disorder. Learn more about the common symptoms of bipolar depression that some patients may experience. Home \u00bb Trending Health News \u00bb 10 Warning Signs of Bipolar Disorder: Depression. One of the most serious symptoms of bipolar disorder is. Bi Polar Depression. SEVERE SWINGS What is bipolar disorder, is it the same as manic depression, what are the symptoms and is there a cure? Bipolar disorder, or manic depression, causes symptoms of mania and depression. Read about bipolar disorder treatment, medications, and causes of this. Learn more about the different types of bipolar disorder. Find out"
                }
            ]
        },
        {
            "question": "How do student loans affect getting a mortgage?",
            "answer": "When applying for a mortgage, student loans can affect the debt to income ratio, which is a key factor in determining the amount that an individual can afford to pay for the mortgage [1]. While student loan repayments do not appear in an individual's credit history and do not affect credit scores, lenders do consider the amount of an individual's student loan repayments when assessing their mortgage application [1][2][3]. Some 83% of non-homeowners say student loan debt is preventing them from buying a home, according to the National Association of Realtors [2]. It is important to note that student loans do not prevent an individual from getting a mortgage [1].",
            "docs": [
                {
                    "title": "Student Loans \u2013 How do they work? | The Financial Review",
                    "text": "typical debt. Student loan repayments do not appear in an individual\u2019s credit history, therefore there are no implications whatsoever. This also extends to applications for credit cards \u2013 student \u2018loans\u2019 are not acknowledged. One noteworthy aspect that is affected by student loans however, is mortgage applications. Nevertheless, it does not prevent an individual from getting a mortgage. For example, lenders will consider the amount of an individual\u2019s student loan repayments in order to assess the debt to income ratio and therefore establish the amount that the individual can afford to pay for the mortgage. Just as they do with other"
                },
                {
                    "title": "How Does Student Loan Debt Affect Buying a Home? | Experian",
                    "text": "Rates & Affordability How Student Loans Affect Getting a Mortgage Student Loan Impact on Credit Scores Other Factors for Getting Approved for a Mortgage If you're a recent college grad and hope to become a homeowner in the near future, you should know that student loan debt could affect buying a home by making it more difficult to get a mortgage. Some 83% of non-homeowners say student loan debt is preventing them from buying a home, according to the National Association of Realtors (NAR). But while student loan payments can make it harder to save for a down payment on"
                },
                {
                    "title": "Studentloanify - How your student loans affect your home mortgage prospects",
                    "text": "Though it may not seem fair, your student loan situation impacts your home mortgage outlook. Many people carry student loan debt, but it\u2019s the amount of the loan and how you handle your student loan repayment plan that will influence your ability to get a home mortgage as well as what your interest rate will be. Here are some specific factors about your student loan that will affect your home mortgage prospects. On your mortgage loan application, you will have to report how much your monthly student loan payment is. This amount will be deducted from your monthly gross income"
                },
                {
                    "title": "How do student loans affect your credit score? | Student Loan Planner",
                    "text": "How do student loans affect your credit score? | Student Loan Planner Your credit score is the three-digit number that dictates a lot in your adult life. Whether you\u2019re applying for a mortgage or looking to get an auto loan, this seemingly arbitrary number determines whether you get approved for a loan and also affects your interest rate. If you\u2019re a student loan borrower you may wonder, \u201cDo student loans affect credit score?\u201d You might be especially curious if you\u2019re in the process of applying for a mortgage. Here\u2019s how student loans affect your credit score and what to know for big life events, like getting a mortgage. Do student loans affect"
                },
                {
                    "title": "Does Student Loan Debt Affect Getting A Mortgage?",
                    "text": "Does Student Loan Debt Affect Getting A Mortgage? Home \u00bb Does Student Loan Debt Affect Getting A Mortgage? Last year, I helped answer a reader\u2019s question about applying for a mortgage while on Income Based Repayment. However, over the last several months, I\u2019ve been getting bombarded with questions about how student loan debt impacts your ability to get a mortgage. Maybe it\u2019s because the housing market is improving, or maybe it\u2019s because people are finally taking their student loan debt seriously. Anyway, I wanted to share a few reader questions and then look at whether student loan debt affects getting a mortgage. Here are the reader questions I\u2019ve"
                }
            ]
        }
    ],
"demos_citation_rag_refusal": [
        {
            "question": "Why did New York City try to ban food donations to the poor?",
            "answer": "New York City, under Mayor Michael Bloomberg's administration, banned citizens from donating food directly to homeless shelters because the city could not assess the salt, fat, and fiber content [1][2][3]. Bloomberg's administration was heavily criticized for losing their common sense by becoming too focused on what people eat [2].",
            "docs": [
                {
                    "title": "The Future Of America",
                    "text": "believe that they are \u201chelping\u201d the homeless by passing such laws. In New York City, Mayor Bloomberg has banned citizens from donating food directly to homeless shelters and he is actually convinced that it was the right thing to do for the homeless\u2026 Mayor Michael Bloomberg\u2019s food police have struck again! Outlawed are food donations to homeless shelters because the city can\u2019t assess their salt, fat and fiber content, reports CBS 2\u2019s Marcia Kramer. Glenn Richter arrived at a West Side synagogue on Monday to collect surplus bagels \u2014 fresh nutritious bagels \u2014 to donate to the poor."
                },
                {
                    "title": "mayor bloomberg",
                    "text": "Amuck: Bloomberg Bans Food Donations in New York City Food Might Be Salty or Too High in Calories, City Explains Washington, D.C. \u2013 New York Mayor Michael Bloomberg\u2019s administration is now banning all food being offered to the city\u2019s homeless shelters. New York City\u2019s bureaucrats have become so singularly focused on what people eat, says the National Center for Public Policy Research, that they\u2019ve lost their common sense. \u201cSo much for serving the homeless: The Bloomberg administration is now taking the term \u2018food police\u2019 to new depths, blocking food donations to all government-run facilities that serve the"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND Front Page Health U.S. New York City bans food donations Inability to control 'nutritional content' cited as reason New York City homeless shelters have Mayor Michael Bloomberg to thank for a halt in food donations, for which hungry families are waiting, according to one public policy advocate. \"The Bloomberg administration is now taking the term 'food police' to new depths, blocking food donations to all government-run facilities that serve the city's homeless,\" says Jeff Stier, a National Center for Public Policy Research senior fellow. Currently, no food can be given to government-run, New York City facilities, despite hungry crowds perfectly"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND Services didn't return WND calls. Stier told WND that he specifically was told by Diamond that the policy was tied to the nutritional guidelines set by the mayor. \"They can say that this ban on donations is a long-standing policy, but they can\u2019t document it,\" Stier told WND. \"I've also been told that there are numerous food shelves that have been accepting food donations, not just one.\" Stier is a member of a New York Synagogue that has donated food for over a decade. He is outraged that the DHS' response to his demand to know why the practice can"
                },
                {
                    "title": "New York City bans food donations - WND",
                    "text": "New York City bans food donations - WND ban on donated food. In fact, it thrives because of food donations. New York City Rescue Mission has been providing food, clothing, shelter and spiritual hope for needy New Yorkers since 1872. \"We feed over 500 people a day, all through donations,\" said James Varnhagen, NYCRM director. \"Boxed food, canned food, prepared food, we take any food,\" he told WND. \"We couldn't survive without donations,\" he said."
                }
            ]
        },
        {
            "question": "What's the difference between Shia vs. Sunni Islam?",
            "answer": "The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership [1]. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. [1][2]. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings, while the Shia branch follows Prophet Muhammad's son-in-law Ali [2]. Nowadays, Sunni and Shia are the major branches of Islam [3].",
            "docs": [
                {
                    "title": "The Sunni vs Shia Divide - Explained - Globaloi",
                    "text": "centuries-long strained relationship between Sunnis and Shias. As a scholar of Islam and a public educator, I often field questions about Sunnis, Shias and the sects of Islam. What exactly is the Shia-Sunni divide? And what is its history? History of divide Both Sunnis and Shias \u2013 drawing their faith and practice from the Qur\u2019an and the life of the Prophet Muhammad \u2013 agree on most of the fundamentals of Islam. The differences are related more to historical events, ideological heritage and issues of leadership. The first and central difference emerged after the death of Prophet Muhammad in A.D. 632."
                },
                {
                    "title": "What\u2019s the difference between Sunni and Shia Islam? \u2013 Macrosnaps",
                    "text": "What\u2019s the difference between Sunni and Shia Islam? Sunni and Shia identities (the 2 main branches of Islam) first formed around a dispute over leadership succession after the death of the Prophet Muhammad in 632 A.D. Sunni is the larger branch (estimated 85-90% of total world Muslim population) and it's adherents are referred to as \"people of the tradition of Muhammad\", while Shia are \"followers\" of Muhammad's son-in-law and cousin Ali. Sunnis rely heavily on the practice of the Prophet Muhammad and his teachings, the Shia view their ayatollahs as reflections of God on earth. What challenges does the anti-IS"
                },
                {
                    "title": "Difference between Sunni and Shia Muslims | Sunni vs Shia Muslims",
                    "text": "of Muhammad, the last prophet of God. A follower of Islam is known as a Muslim. Many Muslims believe that their sole purpose is to worship and serve God, for which they have established five pillars of Islam that guides a Muslim on almost every aspect of life and society. Due to differences, Muslims have been divided into two primary sects: The Sunnis and the Shias. These two sects have many similarities and both consider themselves are Muslims, following the will of God. However, they are also different from each other in certain aspects. Both the Sunnis and the Shias,"
                },
                {
                    "title": "What is the difference between Shia and Sunni Islam? - Islam Stack Exchange",
                    "text": "What is the difference between Shia and Sunni Islam? - Islam Stack Exchange between Mutah marriage and Misyar marriage? What theological and historical factors distinguish Ibadi Islam from either Shia or Sunni schools? What are the principle/fundamental differences between Sunni and Shia? Nikah between a Sunni girl and Shia boy What is the difference between \u201cMubtalat-of-Wudu\u201d of Shia and Sunni? How can the Hadith be reliable when Sunnis and Shia follow different points of reference? Rejection of Mutawatir Hadith in Sunni Islam and Shia Islam"
                },
                {
                    "title": "What is the difference between Sunni and Shia Islam? | Patrick Syder Travel",
                    "text": "What is the difference between Sunni and Shia Islam? | Patrick Syder Travel What is the difference between Sunni and Shia Islam? This Channel 4 link answers some of the key questions about the difference between Sunni and Shia Islam and alarmingly, the politics on what is happening and why, in Syria\u2026\u2026. http://www.channel4.com/news/sunni-shia-islam-muslim-syria-middle-east-key-questions \u2190 Ethiopia Appeal \u2013 Help sponsor a nurse to train and to help others G\u00f6bekli Tepe, Turkey: a new wonder of the ancient world by Jeremy Seal (Telegraph Travel Section 23/04/2013) \u2192"
                }
            ]
        },
{
    "question": "What is the role of nanotechnology in medieval architecture?",
    "answer": "Provided search results do not contain sufficient information to answer the question.",
    "docs": [
        {
            "title": "Medieval Castles: Construction and Design",
            "text": "Medieval castles were remarkable feats of engineering. Built primarily between the 9th and 16th centuries, they were constructed using local stone, wood, and mortar. The design of castles evolved over time to provide increased protection against invaders, featuring thick walls, moats, and fortified towers. Stonemasons used traditional tools and techniques to carve blocks, while lime mortar helped bind the stones together. The architecture reflected both functional and aesthetic considerations, but innovations were limited to the materials and methods available at the time."
        },
        {
            "title": "Nanotechnology: The Science of the Small",
            "text": "Nanotechnology is the manipulation and application of materials at the nanoscale, typically below 100 nanometers. This field has enabled breakthroughs in areas such as medicine, electronics, and materials science. Modern nanotechnology involves the precise engineering of structures and devices at the atomic or molecular level, which can result in unique properties not found in bulk materials. Examples include nanoparticles for drug delivery, carbon nanotubes for strong, lightweight materials, and nanoscale sensors."
        },
        {
            "title": "Building Materials in the Middle Ages",
            "text": "In medieval Europe, common building materials included stone, timber, clay, and lime. Techniques varied regionally depending on available resources. The walls of cathedrals and castles were often made of thick stone to withstand sieges, while wooden beams supported roofs. There were no synthetic or advanced composite materials, and the concept of manipulating matter at the microscopic or atomic scale did not exist."
        },
        {
            "title": "Applications of Nanotechnology in Modern Construction",
            "text": "In contemporary architecture and engineering, nanotechnology is revolutionizing construction. For example, nanoparticles can be added to concrete to enhance its strength and durability, and nanocoatings are used to provide surfaces with self-cleaning or anti-microbial properties. Researchers are exploring nanomaterials to improve energy efficiency in buildings. However, these innovations are very recent and are not part of historical construction methods."
        },
        {
            "title": "Technological Innovations in Medieval Times",
            "text": "The medieval period saw significant advancements in technology, such as the development of the pointed arch, flying buttress, and improved masonry techniques. Tools like the mason's square and the use of scaffolding allowed for more ambitious structures, including Gothic cathedrals. Although these innovations transformed architecture, they were based on macroscopic principles and did not involve atomic or molecular engineering."
        }
    ]
}
    ]
}


hotpotqa_medium = {
    "instruction": "Solve the question step by step and  Integrate the intermediate answers logically to determine the final answer. ",
    "instruction_citation": "Solve the question step by step and integrate the intermediate answers logically to determine the final answer. For each fact claim, cite the source. ",
    "instruction_citation2": "Solve the question step by step and integrate the intermediate answers logically to determine the final answer. For every factual claim, cite the source immediately after the claim using the following format: <|reserved_special_token_0|> title <|reserved_special_token_1|>, where title is the title of the document that supports the claim.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    # "doc_prompt": "Document: {P}\n",
    "doc_prompt": "Document [{ID}](Title: {T}): {P}\n\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Are Malcolm St. Clair and Nicolas Roeg both actors?",
            "answer": "Malcolm St. Clair is a Hollywood film director, writer, producer and actor. Nicolas Roeg is an English film director and cinematographer. So, Malcolm St. Clair is an actor, but Nicolas Roeg is not an actor. So the answer is no.",
            "docs": [
                {
                    "title": "Malcolm St. Clair (filmmaker)",
                    "text": 'Malcolm St. Clair (May 17, 1897 – June 1, 1952), also known as Mal St Clair, was a Hollywood film director, writer, producer and actor.'
                },
                {
                    "title": "Nicolas Roeg",
                    "text": 'Nicolas Jack Roeg (born 15 August 1928) is an English film director and cinematographer.'
                }
            ]
        },
        {
            "question": "Which state is the maker of Turbo Drop located in?",
            "answer": "The maker of Turbo Drop is S&S Worldwide. And the location of S&S Worldwide is Utah. So the answer is Utah.",
            "docs": [
                {
                    "title": "Turbo Drop",
                    "text": "Turbo Drop is a type of tower-based amusement ride manufactured by S&S Worldwide, similar to the company's Space Shot.  The current tallest Turbo Drop installation in operation is Supreme Scream at Knott's Berry Farm in Buena Park, California."
                },
                {
                    "title": "Supreme Scream",
                    "text": "Supreme Scream is a drop tower located at Knott's Berry Farm in Buena Park, CA, USA and was designed and manufactured by S&S Worldwide, Inc. of Utah."
                },
            ]
        },
        {
            "question": "What county does the  Delphus Kill flows into the Mohawk River?",
            "answer": "The Delphus Kill flows into the Mohawk River in Dunsbach Ferry. Dunsbach Ferry is located in the town of Colonie, which is in Albany County, New York. So the answer is Albany County.",
            "docs": [
                {
                    "title": "Delphus Kill",
                    "text": "The Delphus Kill flows into the Mohawk River in Dunsbach Ferry, New York."
                },
                {
                    "title": "Dunsbach Ferry, New York",
                    "text": "Dunsbach Ferry is a hamlet of the town of Colonie, in Albany County, New York."
                }
            ]
        },

    ],
    "demos_citation": [
        {
            "question": "Are Malcolm St. Clair and Nicolas Roeg both actors?",
            # "answer": "Malcolm St. Clair is a Hollywood film director, writer, producer and actor [1] <Malcolm St. Clair (filmmaker)>. Nicolas Roeg is an English film director and cinematographer [2] <Nicolas Roeg>. So, Malcolm St. Clair is an actor, but Nicolas Roeg is not an actor. So the answer is no.",
            "answer": "Malcolm St. Clair is a Hollywood film director, writer, producer and actor. <source> Malcolm St. Clair (filmmaker) </source>. Nicolas Roeg is an English film director and cinematographer. <source> Nicolas Roeg </source>. So, Malcolm St. Clair is an actor, but Nicolas Roeg is not an actor. So the answer is no.",
            "docs": [
                {
                    "title": "Malcolm St. Clair (filmmaker)",
                    "text": 'Malcolm St. Clair (May 17, 1897 – June 1, 1952), also known as Mal St Clair, was a Hollywood film director, writer, producer and actor.'
                },
                {
                    "title": "Nicolas Roeg",
                    "text": 'Nicolas Jack Roeg (born 15 August 1928) is an English film director and cinematographer.'
                }
            ]
        },
        {
            "question": "Which state is the maker of Turbo Drop located in?",
            # "answer": "The maker of Turbo Drop is S&S Worldwide [1] <Turbo Drop>. And the location of S&S Worldwide is Utah [2] <Supreme Scream>. So the answer is Utah.",
            "answer": "The maker of Turbo Drop is S&S Worldwide <source> Turbo Drop </source>. And the location of S&S Worldwide is Utah <source> Supreme Scream </source>. So the answer is Utah.",
            "docs": [
                {
                    "title": "Turbo Drop",
                    "text": "Turbo Drop is a type of tower-based amusement ride manufactured by S&S Worldwide, similar to the company's Space Shot.  The current tallest Turbo Drop installation in operation is Supreme Scream at Knott's Berry Farm in Buena Park, California."
                },
                {
                    "title": "Supreme Scream",
                    "text": "Supreme Scream is a drop tower located at Knott's Berry Farm in Buena Park, CA, USA and was designed and manufactured by S&S Worldwide, Inc. of Utah."
                },
            ]
        },
        {
            "question": "What county does the  Delphus Kill flows into the Mohawk River?",
            # "answer": "The Delphus Kill flows into the Mohawk River in Dunsbach Ferry [1] <Delphus Kill>. Dunsbach Ferry is located in the town of Colonie, which is in Albany County, New York [2] <Dunsbach Ferry, New York>. So the answer is Albany County.",
            "answer": "The Delphus Kill flows into the Mohawk River in Dunsbach Ferry <source> Delphus Kill </source>. Dunsbach Ferry is located in the town of Colonie, which is in Albany County, New York <source> Dunsbach Ferry, New York </source>. So the answer is Albany County.",
            "docs": [
                {
                    "title": "Delphus Kill",
                    "text": "The Delphus Kill flows into the Mohawk River in Dunsbach Ferry, New York."
                },
                {
                    "title": "Dunsbach Ferry, New York",
                    "text": "Dunsbach Ferry is a hamlet of the town of Colonie, in Albany County, New York."
                }
            ]
        },
    ],
    "demos_citation_st": [
        {
            "question": "Are Malcolm St. Clair and Nicolas Roeg both actors?",
            "answer": f"Malcolm St. Clair is a Hollywood film director, writer, producer and actor. {SOT_TOKEN}Malcolm St. Clair (filmmaker){EOT_TOKEN}. Nicolas Roeg is an English film director and cinematographer. {SOT_TOKEN}Nicolas Roeg{EOT_TOKEN}. So, Malcolm St. Clair is an actor, but Nicolas Roeg is not an actor. So the answer is no.",
            "docs": [
                {
                    "title": "Malcolm St. Clair (filmmaker)",
                    "text": 'Malcolm St. Clair (May 17, 1897 – June 1, 1952), also known as Mal St Clair, was a Hollywood film director, writer, producer and actor.'
                },
                {
                    "title": "Nicolas Roeg",
                    "text": 'Nicolas Jack Roeg (born 15 August 1928) is an English film director and cinematographer.'
                }
            ]
        },
        {
            "question": "Which state is the maker of Turbo Drop located in?",
            "answer": f"The maker of Turbo Drop is S&S Worldwide. {SOT_TOKEN}Turbo Drop{EOT_TOKEN}. And the location of S&S Worldwide is Utah. {SOT_TOKEN}Supreme Scream{EOT_TOKEN}. So the answer is Utah.",
            "docs": [
                {
                    "title": "Turbo Drop",
                    "text": "Turbo Drop is a type of tower-based amusement ride manufactured by S&S Worldwide, similar to the company's Space Shot.  The current tallest Turbo Drop installation in operation is Supreme Scream at Knott's Berry Farm in Buena Park, California."
                },
                {
                    "title": "Supreme Scream",
                    "text": "Supreme Scream is a drop tower located at Knott's Berry Farm in Buena Park, CA, USA and was designed and manufactured by S&S Worldwide, Inc. of Utah."
                },
            ],
        },
        {
            "question": "What county does the  Delphus Kill flows into the Mohawk River?",
            "answer": f"The Delphus Kill flows into the Mohawk River in Dunsbach Ferry. {SOT_TOKEN}Delphus Kill{EOT_TOKEN}. Dunsbach Ferry is located in the town of Colonie, which is in Albany County, New York. {SOT_TOKEN}Dunsbach Ferry, New York{EOT_TOKEN}. So the answer is Albany County.",
            "docs": [
                {
                    "title": "Delphus Kill",
                    "text": "The Delphus Kill flows into the Mohawk River in Dunsbach Ferry, New York."
                },
                {
                    "title": "Dunsbach Ferry, New York",
                    "text": "Dunsbach Ferry is a hamlet of the town of Colonie, in Albany County, New York."
                }
            ]
        },
    ]
}

repliqa = {
    "instruction": "Provide a clear and concise answer for the following question. ",
    "instruction_citation": "Provide a concise answer for the following question and cite the source.",
    "instruction_citation2": "Provide a concise answer for the following question and give one citation at the end of your answer. The citation should be in the format of <|reserved_special_token_0|> title <|reserved_special_token_1|>, where title is the title of the document that supports your answer.",
    "instruction_citation_rag": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
    "instruction_citation_rag_refusal": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. If the provided search results do not contain sufficient information to answer the question, respond with \"Provided search results do not contain sufficient information to answer the question.\"",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document [{ID}](Title: {T}): {P}\n\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What is the main benefit of government incentives and programs for SMEs?",
            "answer": "Reducing barriers to expansion for SMEs.",
            "docs": [
                {
                    "title": "Funding and Investment Strategies for SME Expansion: Navigating New Horizons Beyond 2023",
                    "text": 'Government bodies worldwide recognize the pivotal role of SMEs in economic growth and are launching various programs and incentives to support them. Tax breaks, reduced interest rates for business loans, and easier compliance requirements are being rolled out to aid SME expansion.'
                }
            ]
        },
        {
            "question": "What sustainable practices will be implemented at the Green Soundscape festival?",
            "answer": "The Green Soundscape festival will feature recycling stations, compostable materials, and solar-powered stages.",
            "docs": [
                {
                    "title": "Local Music Festivals and Events to Watch: A Deep Dive into Logistics",
                    "text": "No discussion about festival logistics is complete without addressing the environmental impact. The influx of festivals like the three-day Green Soundscape, scheduled for September 22-24, brings sustainability to the forefront. Spearheaded by environmentalist and event coordinator Jonas Schmidt, Green Soundscape promises a lineup of alternative and indie artists, coupled with a zero-waste policy. 'A challenge we face every year is reconciling the desire for an immersive festival experience with our environmental responsibilities,’ Schmidt explains. Recycling stations, compostable materials, and solar-powered stages are just a few measures implemented. Yet, the endeavor extends further into transportation logistics, encouraging festival-goers to use public transit, bikes, and other eco-friendly travel methods."
                },
            ]
        },
        {
            "question": "What method has 'Verde Vista' utilized to secure funding despite the decrease in public grants?",
            "answer": "Crowdfunding and partnerships with eco-conscious companies.",
            "docs": [
                {
                    "title": "Eco-Art Movements: Environmentally Focused Art Festivals Embrace a Greener Vision",
                    "text": "Funding green initiatives presents its own set of challenges. Public grants for the arts have been shrinking, and environmentally-focused projects often compete with a plethora of other social causes. However, festivals like 'Verde Vista' have found success through crowdfunding campaigns and partnerships with eco-conscious companies. 'It's not just about finding sponsors but aligning with organizations that genuinely care about their carbon footprint and societal impact,' states Torres. Patrons of the arts are also progressively gravitating towards supporting causes that have meaningful impacts beyond the art world."
                }
            ]
        },

    ],
    "demos_citation": [
        {
            "question": "What is the main benefit of government incentives and programs for SMEs?",
            "answer": "Reducing barriers to expansion for SMEs. [1] Funding and Investment Strategies for SME Expansion: Navigating New Horizons Beyond 2023",
            "docs": [
                {
                    "title": "Funding and Investment Strategies for SME Expansion: Navigating New Horizons Beyond 2023",
                    "text": 'Government bodies worldwide recognize the pivotal role of SMEs in economic growth and are launching various programs and incentives to support them. Tax breaks, reduced interest rates for business loans, and easier compliance requirements are being rolled out to aid SME expansion.'
                },
            ]
        },
        {
            "question": "What sustainable practices will be implemented at the Green Soundscape festival?",
            "answer": "The Green Soundscape festival will feature recycling stations, compostable materials, and solar-powered stages. [1] Local Music Festivals and Events to Watch: A Deep Dive into Logistics",
            "docs": [
                {
                    "title": "Local Music Festivals and Events to Watch: A Deep Dive into Logistics",
                    "text": "No discussion about festival logistics is complete without addressing the environmental impact. The influx of festivals like the three-day Green Soundscape, scheduled for September 22-24, brings sustainability to the forefront. Spearheaded by environmentalist and event coordinator Jonas Schmidt, Green Soundscape promises a lineup of alternative and indie artists, coupled with a zero-waste policy. 'A challenge we face every year is reconciling the desire for an immersive festival experience with our environmental responsibilities,’ Schmidt explains. Recycling stations, compostable materials, and solar-powered stages are just a few measures implemented. Yet, the endeavor extends further into transportation logistics, encouraging festival-goers to use public transit, bikes, and other eco-friendly travel methods."
                },
            ]
        },
        {
            "question": "What method has 'Verde Vista' utilized to secure funding despite the decrease in public grants?",
            "answer": "Crowdfunding and partnerships with eco-conscious companies. [1] Eco-Art Movements: Environmentally Focused Art Festivals Embrace a Greener Vision",
            "docs": [
                {
                    "title": "Eco-Art Movements: Environmentally Focused Art Festivals Embrace a Greener Vision",
                    "text": "Funding green initiatives presents its own set of challenges. Public grants for the arts have been shrinking, and environmentally-focused projects often compete with a plethora of other social causes. However, festivals like 'Verde Vista' have found success through crowdfunding campaigns and partnerships with eco-conscious companies. 'It's not just about finding sponsors but aligning with organizations that genuinely care about their carbon footprint and societal impact,' states Torres. Patrons of the arts are also progressively gravitating towards supporting causes that have meaningful impacts beyond the art world."
                }
            ]
        },

    ],
"demos_citation_rag": [
    {
        "question": "What is the main benefit of government incentives and programs for SMEs?",
        "answer": "Reducing barriers to expansion for SMEs is a main benefit of government incentives and programs, which often include tax breaks, reduced loan interest rates, and simplified compliance processes [2].",
        "docs": [
            {
                "title": "The Rise of Remote Work: Impacts on Urban Business Zones",
                "text": "Remote work has changed commercial real estate dynamics, leading to reduced demand for SME office spaces in city centers. This shift affects foot traffic but is unrelated to government incentive programs."
            },
            {
                "title": "Funding and Investment Strategies for SME Expansion: Navigating New Horizons Beyond 2023",
                "text": "Government bodies worldwide recognize the pivotal role of SMEs in economic growth and are launching various programs and incentives to support them. Tax breaks, reduced interest rates for business loans, and easier compliance requirements are being rolled out to aid SME expansion."
            },
            {
                "title": "Understanding Corporate ESG Reporting Obligations",
                "text": "Large enterprises face growing pressure to meet environmental, social, and governance (ESG) standards. SMEs are often exempt or face lighter requirements, making this topic adjacent but not focused on expansion incentives."
            },
            {
                "title": "E-Commerce Boom Drives Packaging Innovation",
                "text": "The growth of e-commerce has led companies to invest in sustainable and flexible packaging solutions. While this trend involves many SMEs, it does not directly relate to tax or regulatory incentives."
            },
            {
                "title": "How Blockchain Is Disrupting Supply Chain Transparency",
                "text": "Blockchain adoption is accelerating across supply chains, especially among logistics providers. Although SMEs may benefit indirectly from transparent sourcing, the technology primarily serves larger logistics networks."
            }
        ]
    },
    {
        "question": "What sustainable practices will be implemented at the Green Soundscape festival?",
        "answer": "The Green Soundscape festival will implement sustainable practices such as recycling stations, compostable materials, and solar-powered stages [5].",
        "docs": [
            {
                "title": "Rain Delays Expected to Impact Weekend Concerts",
                "text": "Heavy rainfall is forecasted to hit the region this weekend, leading to possible cancellations and venue changes for several outdoor music events. Organizers are considering tenting options."
            },
            {
                "title": "The Evolution of Sound Design in Ambient Music",
                "text": "Recent trends in ambient music emphasize layered soundscapes and analog synths. While this genre may be performed at eco-themed festivals, this topic doesn’t address sustainability measures."
            },
            {
                "title": "Behind the Scenes: Artist Hospitality at Major Festivals",
                "text": "Organizing green rooms and hospitality suites is a logistical feat. While artists at Green Soundscape may benefit from eco-conscious amenities, this does not describe sustainability practices for the public."
            },
            {
                "title": "Food Trucks at Festivals: A Growing Business Trend",
                "text": "Mobile food vendors are becoming key features at outdoor festivals. Popular offerings include fusion street food, vegan wraps, and nitro cold brew, but food practices are not the focus here."
            },
            {
                "title": "Local Music Festivals and Events to Watch: A Deep Dive into Logistics",
                "text": "No discussion about festival logistics is complete without addressing the environmental impact. The influx of festivals like the three-day Green Soundscape, scheduled for September 22–24, brings sustainability to the forefront. Spearheaded by environmentalist and event coordinator Jonas Schmidt, Green Soundscape promises a lineup of alternative and indie artists, coupled with a zero-waste policy. Recycling stations, compostable materials, and solar-powered stages are just a few measures implemented."
            }
        ]
    },
    {
        "question": "What method has 'Verde Vista' utilized to secure funding despite the decrease in public grants?",
        "answer": "'Verde Vista' has secured funding despite a decrease in public grants by leveraging crowdfunding campaigns and forming partnerships with eco-conscious companies [1].",
        "docs": [
            {
                "title": "Eco-Art Movements: Environmentally Focused Art Festivals Embrace a Greener Vision",
                "text": "Funding green initiatives presents its own set of challenges. Public grants for the arts have been shrinking, and environmentally-focused projects often compete with other causes. However, festivals like 'Verde Vista' have found success through crowdfunding campaigns and partnerships with eco-conscious companies."
            },
            {
                "title": "Volunteerism in Community Theater: Challenges and Rewards",
                "text": "Community theaters rely on volunteers for staffing, direction, and even minor funding support. Their challenges differ significantly from large-scale eco-art festivals like Verde Vista."
            },
            {
                "title": "Trends in Paint Recycling and Waste Management",
                "text": "Recycling latex and oil-based paints has become a municipal focus in several cities. While sustainability is a common theme, these initiatives are not tied to arts funding or crowdfunding."
            },
            {
                "title": "The Carbon Cost of Touring Art Exhibits",
                "text": "Art exhibits that travel internationally face scrutiny for their carbon emissions. Though related to environmentalism in art, this topic doesn’t cover financial sourcing or partnerships."
            },
            {
                "title": "How Film Festivals Are Embracing Digital Distribution",
                "text": "As streaming becomes dominant, many indie film festivals are choosing to premiere online. This shift affects budgets and outreach strategies but does not relate to art festival financing models."
            }
        ]
    }
],
"demos_citation_rag_refusal": [
    {
        "question": "What is the main benefit of government incentives and programs for SMEs?",
        "answer": "Reducing barriers to expansion for SMEs is a main benefit of government incentives and programs, which often include tax breaks, reduced loan interest rates, and simplified compliance processes [2].",
        "docs": [
            {
                "title": "The Rise of Remote Work: Impacts on Urban Business Zones",
                "text": "Remote work has changed commercial real estate dynamics, leading to reduced demand for SME office spaces in city centers. This shift affects foot traffic but is unrelated to government incentive programs."
            },
            {
                "title": "Funding and Investment Strategies for SME Expansion: Navigating New Horizons Beyond 2023",
                "text": "Government bodies worldwide recognize the pivotal role of SMEs in economic growth and are launching various programs and incentives to support them. Tax breaks, reduced interest rates for business loans, and easier compliance requirements are being rolled out to aid SME expansion."
            },
            {
                "title": "Understanding Corporate ESG Reporting Obligations",
                "text": "Large enterprises face growing pressure to meet environmental, social, and governance (ESG) standards. SMEs are often exempt or face lighter requirements, making this topic adjacent but not focused on expansion incentives."
            },
            {
                "title": "E-Commerce Boom Drives Packaging Innovation",
                "text": "The growth of e-commerce has led companies to invest in sustainable and flexible packaging solutions. While this trend involves many SMEs, it does not directly relate to tax or regulatory incentives."
            },
            {
                "title": "How Blockchain Is Disrupting Supply Chain Transparency",
                "text": "Blockchain adoption is accelerating across supply chains, especially among logistics providers. Although SMEs may benefit indirectly from transparent sourcing, the technology primarily serves larger logistics networks."
            }
        ]
    },
    {
         "question": "How did the 'Solar Weave' textile project contribute to advancements in biodegradable energy storage?",
    "answer": "Provided search results do not contain sufficient information to answer the question.",
    "docs": [
        {
            "title": "Biodegradable Materials in Fashion: The Next Wave",
            "text": "Recent years have seen a surge in biodegradable materials used in the fashion industry, such as natural fibers and plant-based dyes. Designers are now prioritizing eco-friendly materials to reduce environmental impact. These innovations are paving the way for more sustainable production cycles in both high fashion and mass-market apparel. However, most developments focus on wearability and end-of-life disposal rather than integration with energy systems."
        },
        {
            "title": "Smart Textiles: Trends in Wearable Technology",
            "text": "Smart textiles incorporate electronics or responsive materials to enhance clothing functionality. Applications include temperature regulation, activity monitoring, and LED-embedded garments. Some experimental fabrics have explored energy storage through embedded batteries, though these often use traditional non-biodegradable components. Ongoing research aims to improve washability, safety, and efficiency of wearable tech."
        },
        {
            "title": "Solar Weave: Reimagining Solar Panels for Urban Rooftops",
            "text": "The Solar Weave project is an architectural innovation that adapts flexible solar panels into woven mesh for rooftop installations. By interlacing photovoltaic strips into a durable polymer lattice, Solar Weave aims to maximize solar capture in dense urban environments. The project has been lauded for improving efficiency and flexibility in solar deployment. However, the main focus is on renewable energy generation for buildings, not textile or energy storage applications."
        },
        {
            "title": "Advancements in Biodegradable Batteries",
            "text": "Engineers are developing biodegradable batteries composed of materials like cellulose, silk proteins, and zinc. These next-generation batteries could power medical implants or disposable electronics, reducing e-waste. Some prototypes dissolve safely in water or soil after use. The research is ongoing, and while promising, these batteries are not yet widely adopted outside laboratory settings."
        },
        {
            "title": "Energy Storage Solutions for Off-Grid Applications",
            "text": "Reliable energy storage is crucial for off-grid solar, wind, and microgrid systems. Current technologies include lithium-ion, flow, and sodium-sulfur batteries, as well as thermal storage. New research is exploring ways to make these systems more sustainable and cost-effective. However, integration with textiles or clothing has not been a major area of study in this context."
        }
    ]
    },
    {
        "question": "What method has 'Verde Vista' utilized to secure funding despite the decrease in public grants?",
        "answer": "'Verde Vista' has secured funding despite a decrease in public grants by leveraging crowdfunding campaigns and forming partnerships with eco-conscious companies [1].",
        "docs": [
            {
                "title": "Eco-Art Movements: Environmentally Focused Art Festivals Embrace a Greener Vision",
                "text": "Funding green initiatives presents its own set of challenges. Public grants for the arts have been shrinking, and environmentally-focused projects often compete with other causes. However, festivals like 'Verde Vista' have found success through crowdfunding campaigns and partnerships with eco-conscious companies."
            },
            {
                "title": "Volunteerism in Community Theater: Challenges and Rewards",
                "text": "Community theaters rely on volunteers for staffing, direction, and even minor funding support. Their challenges differ significantly from large-scale eco-art festivals like Verde Vista."
            },
            {
                "title": "Trends in Paint Recycling and Waste Management",
                "text": "Recycling latex and oil-based paints has become a municipal focus in several cities. While sustainability is a common theme, these initiatives are not tied to arts funding or crowdfunding."
            },
            {
                "title": "The Carbon Cost of Touring Art Exhibits",
                "text": "Art exhibits that travel internationally face scrutiny for their carbon emissions. Though related to environmentalism in art, this topic doesn’t cover financial sourcing or partnerships."
            },
            {
                "title": "How Film Festivals Are Embracing Digital Distribution",
                "text": "As streaming becomes dominant, many indie film festivals are choosing to premiere online. This shift affects budgets and outreach strategies but does not relate to art festival financing models."
            }
        ]
    }
]

}
mhrag = {
    "instruction": "Solve the question step by step and  Integrate the intermediate answers logically to determine the final answer. ",
    "instruction_citation": "Solve the question step by step and integrate the intermediate answers logically to determine the final answer. For each fact claim, cite the source. ",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            'question': "Which company, cited in articles from both The Verge and TechCrunch, is involved in legal disputes for making exclusive deals with major tech companies, defending against accusations in an antitrust battle with a game maker, and facing a class action lawsuit for its impact on news publishers' revenues?",
            'answer': "The company involved in legal disputes for making exclusive deals with major tech companies is Google. The Verge article mentions that Google makes deals with Apple, Samsung, and Mozilla to ensure it is the default search engine on various platforms. TechCrunch highlights Google's antitrust battle with Fortnite maker Epic Games, where Epic suggests Google hid items from discovery. Another TechCrunch article notes a class action antitrust suit against Google by news publisher Helena World Chronicle, citing Google's impact on news publishers' revenues. Therefore, the answer is Google.",
            'docs': [{'id': 'https://www.theverge.com/2023/9/26/23891037/apple-eddy-cue-testimony-us-google',
                      'text': 'The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.',
                      'title': 'The Verge: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’'},
                     {
                         'id': 'https://techcrunch.com/2023/11/06/what-to-know-about-fortnite-maker-epic-games-antitrust-battle-with-google-starting-today/',
                         'text': 'Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.',
                         'title': 'TechCrunch: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today'},
                     {
                         'id': 'https://techcrunch.com/2023/12/15/news-publisher-files-class-action-antitrust-suit-against-google-citing-ais-harms-to-their-bottom-line/',
                         'text': 'The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.',
                         'title': 'TechCrunch: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line'}]},
        {
            'question': 'Does the Sporting News article suggest that the MLB offseason activities will intensify during a specific period, while the CBSSports.com article indicates that Ryan Blaney and other drivers participated in offseason testing, thus showing a difference in the type of offseason activities reported by each source?',
            'answer': 'The Sporting News article suggests that the MLB offseason activities will intensify during a specific period from Sunday, Dec. 3, to Wednesday, Dec. 6. The CBSSports.com article indicates that Ryan Blaney and other drivers participated in offseason testing. Therefore, the articles show a difference in the type of offseason activities reported by each source. So the answer is yes.',
            'docs': [{
                'id': 'https://www.sportingnews.com/us/mlb/news/mlb-winter-meetings-news-signings-trades-2023-free-agency/b6561c0450fe40cdfd39860e',
                'text': "Starting Sunday, Dec. 3, and running through Wednesday, Dec. 6, expect MLB's offseason to really kick into high gear.",
                'title': 'Sporting News: MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency'},
                {
                    'id': 'https://www.cbssports.com/nascar/news/nascar-offseason-news-roundup-short-track-testing-in-phoenix-new-spotters-for-stars-more-moves-for-2024/',
                    'text': 'Blaney was one of six drivers to participate in the test, joining Chris Buescher, Erik Jones, Christopher Bell, Kyle Larson and Corey LaJoie.',
                    'title': 'CBSSports.com: NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024'}]},
        {
            'question': "Does the TechCrunch article reporting on the EU's investigation into X over alleged disinformation disagree with the TechCrunch article warning Elon Musk's X (formerly Twitter) about illegal content and disinformation in terms of the type of content being circulated, with both mentioning the circulation of false information and manipulated content?",
            'answer': "The TechCrunch article reporting on the EU's investigation into X mentions posts identified as false, such as a clip that purported to show Hamas missile attacks on Israel but was actually footage from a video game. The TechCrunch article warning Elon Musk's X about illegal content and disinformation states that there are instances of fake and manipulated images and facts circulating on the platform, such as repurposed old images of unrelated armed conflicts or military footage from video games. Both articles mention the circulation of false information and manipulated content. Therefore, the articles do not disagree about the type of content being circulated. So the answer is no.",
            'docs': [{'id': 'https://techcrunch.com/2023/10/12/eu-x-investigation/',
                      'text': 'Since Saturday’s attacks in Israel, posts identified as false have been spotted circulating on X — including, in one example, a clip that purported to show Hamas missile attacks on Israel but was actually footage from a video game.',
                      'title': 'TechCrunch: After warning, the EU opens investigation into X over alleged disinformation'},
                     {'id': 'https://techcrunch.com/2023/10/10/eu-dsa-warning-elon-musk-x/',
                      'text': '“Public media and civil society organisations widely report instances of fake and manipulated images and facts circulating on your platform in the EU, such as repurposed old images of unrelated armed conflicts or military footage that actually originated from video games.',
                      'title': 'TechCrunch: EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks'}]},

    ],
    "demos_citation": [
        {
            'question': "Which company, cited in articles from both The Verge and TechCrunch, is involved in legal disputes for making exclusive deals with major tech companies, defending against accusations in an antitrust battle with a game maker, and facing a class action lawsuit for its impact on news publishers' revenues?",
            # 'answer': "Google is involved in legal disputes for making exclusive deals with major tech companies, as mentioned in The Verge article about Google Search deals with Apple, Samsung, and Mozilla. [1] <The Verge: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’> Google is defending against accusations in an antitrust battle with game maker Epic Games, as noted in the TechCrunch article about Epic Games' antitrust battle with Google. [2] <TechCrunch: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today> Additionally, Google is facing a class action lawsuit for its impact on news publishers' revenues, as stated in the TechCrunch article about the news publisher's class action antitrust suit against Google. [3] <TechCrunch: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line> Therefore, the company cited in articles from both The Verge and TechCrunch is Google.",
            'answer': "Google is involved in legal disputes for making exclusive deals with major tech companies, as mentioned in The Verge article about Google Search deals with Apple, Samsung, and Mozilla. <source> The Verge: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ </source>. Google is defending against accusations in an antitrust battle with game maker Epic Games, as noted in the TechCrunch article about Epic Games' antitrust battle with Google. <source> TechCrunch: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today </source>. Additionally, Google is facing a class action lawsuit for its impact on news publishers' revenues, as stated in the TechCrunch article about the news publisher's class action antitrust suit against Google. <source> TechCrunch: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line </source>. Therefore, the company cited in articles from both The Verge and TechCrunch is Google.",
            'docs': [{'id': 'https://www.theverge.com/2023/9/26/23891037/apple-eddy-cue-testimony-us-google',
                      'text': 'The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.',
                      'title': 'The Verge: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’'},
                     {
                         'id': 'https://techcrunch.com/2023/11/06/what-to-know-about-fortnite-maker-epic-games-antitrust-battle-with-google-starting-today/',
                         'text': 'Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.',
                         'title': 'TechCrunch: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today'},
                     {
                         'id': 'https://techcrunch.com/2023/12/15/news-publisher-files-class-action-antitrust-suit-against-google-citing-ais-harms-to-their-bottom-line/',
                         'text': 'The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.',
                         'title': 'TechCrunch: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line'}]},
        {
            'question': 'Does the Sporting News article suggest that the MLB offseason activities will intensify during a specific period, while the CBSSports.com article indicates that Ryan Blaney and other drivers participated in offseason testing, thus showing a difference in the type of offseason activities reported by each source?',
            # 'answer': 'The Sporting News article suggests that the MLB offseason activities will intensify during a specific period from Sunday, Dec. 3, to Wednesday, Dec. 6. [1] <Sporting News: MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency> The CBSSports.com article indicates that Ryan Blaney and other drivers participated in offseason testing. [2] <CBSSports.com: NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024> Therefore, the articles show a difference in the type of offseason activities reported by each source. So the answer is yes.',
            'answer': 'The Sporting News article suggests that the MLB offseason activities will intensify during a specific period from Sunday, Dec. 3, to Wednesday, Dec. 6. <source> Sporting News: MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency </source>. The CBSSports.com article indicates that Ryan Blaney and other drivers participated in offseason testing. <source> CBSSports.com: NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 </source>. Therefore, the articles show a difference in the type of offseason activities reported by each source. So the answer is yes.',
            'docs': [{
                         'id': 'https://www.sportingnews.com/us/mlb/news/mlb-winter-meetings-news-signings-trades-2023-free-agency/b6561c0450fe40cdfd39860e',
                         'text': "Starting Sunday, Dec. 3, and running through Wednesday, Dec. 6, expect MLB's offseason to really kick into high gear.",
                         'title': 'Sporting News: MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency'},
                     {
                         'id': 'https://www.cbssports.com/nascar/news/nascar-offseason-news-roundup-short-track-testing-in-phoenix-new-spotters-for-stars-more-moves-for-2024/',
                         'text': 'Blaney was one of six drivers to participate in the test, joining Chris Buescher, Erik Jones, Christopher Bell, Kyle Larson and Corey LaJoie.',
                         'title': 'CBSSports.com: NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024'}]},

        {
            'question': "Does the TechCrunch article reporting on the EU's investigation into X over alleged disinformation disagree with the TechCrunch article warning Elon Musk's X (formerly Twitter) about illegal content and disinformation in terms of the type of content being circulated, with both mentioning the circulation of false information and manipulated content?",
            'answer': "The TechCrunch article reporting on the EU's investigation into X mentions posts identified as false, such as a clip that purported to show Hamas missile attacks on Israel but was actually footage from a video game. <source> TechCrunch: After warning, the EU opens investigation into X over alleged disinformation </source>. The TechCrunch article warning Elon Musk's X about illegal content and disinformation states that there are instances of fake and manipulated images and facts circulating on the platform, such as repurposed old images of unrelated armed conflicts or military footage from video games. <source> TechCrunch: EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks </source>. Both articles mention the circulation of false information and manipulated content. Therefore, the articles do not disagree about the type of content being circulated. So the answer is no.",
            'docs': [{'id': 'https://techcrunch.com/2023/10/12/eu-x-investigation/',
                      'text': 'Since Saturday’s attacks in Israel, posts identified as false have been spotted circulating on X — including, in one example, a clip that purported to show Hamas missile attacks on Israel but was actually footage from a video game.',
                      'title': 'TechCrunch: After warning, the EU opens investigation into X over alleged disinformation'},
                     {'id': 'https://techcrunch.com/2023/10/10/eu-dsa-warning-elon-musk-x/',
                      'text': '“Public media and civil society organisations widely report instances of fake and manipulated images and facts circulating on your platform in the EU, such as repurposed old images of unrelated armed conflicts or military footage that actually originated from video games.',
                      'title': 'TechCrunch: EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks'}]},

    ],
    "demos_citation_st": [
        {
            'question': "Which company, cited in articles from both The Verge and TechCrunch, is involved in legal disputes for making exclusive deals with major tech companies, defending against accusations in an antitrust battle with a game maker, and facing a class action lawsuit for its impact on news publishers' revenues?",
            'answer': "Google is involved in legal disputes for making exclusive deals with major tech companies, as mentioned in The Verge article about Google Search deals with Apple, Samsung, and Mozilla. <|reserved_special_token_0|> The Verge: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ <|reserved_special_token_1|> Google is defending against accusations in an antitrust battle with game maker Epic Games, as noted in the TechCrunch article about Epic Games' antitrust battle with Google. <|reserved_special_token_0|> TechCrunch: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today <|reserved_special_token_1|> Additionally, Google is facing a class action lawsuit for its impact on news publishers' revenues, as stated in the TechCrunch article about the news publisher's class action antitrust suit against Google. <|reserved_special_token_0|> TechCrunch: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line <|reserved_special_token_1|> Therefore, the company cited in articles from both The Verge and TechCrunch is Google.",
            'docs': [{'id': 'https://www.theverge.com/2023/9/26/23891037/apple-eddy-cue-testimony-us-google',
                      'text': 'The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.',
                      'title': 'The Verge: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’'},
                     {
                         'id': 'https://techcrunch.com/2023/11/06/what-to-know-about-fortnite-maker-epic-games-antitrust-battle-with-google-starting-today/',
                         'text': 'Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.',
                         'title': 'TechCrunch: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today'},
                     {
                         'id': 'https://techcrunch.com/2023/12/15/news-publisher-files-class-action-antitrust-suit-against-google-citing-ais-harms-to-their-bottom-line/',
                         'text': 'The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.',
                         'title': 'TechCrunch: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line'}]},
        {
            'question': 'Does the Sporting News article suggest that the MLB offseason activities will intensify during a specific period, while the CBSSports.com article indicates that Ryan Blaney and other drivers participated in offseason testing, thus showing a difference in the type of offseason activities reported by each source?',
            'answer': 'The Sporting News article suggests that the MLB offseason activities will intensify during a specific period from Sunday, Dec. 3, to Wednesday, Dec. 6. <|reserved_special_token_0|> Sporting News: MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency <|reserved_special_token_1|> The CBSSports.com article indicates that Ryan Blaney and other drivers participated in offseason testing. <|reserved_special_token_0|> CBSSports.com: NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 <|reserved_special_token_1|> Therefore, the articles show a difference in the type of offseason activities reported by each source. So the answer is yes.',
            'docs': [{
                         'id': 'https://www.sportingnews.com/us/mlb/news/mlb-winter-meetings-news-signings-trades-2023-free-agency/b6561c0450fe40cdfd39860e',
                         'text': "Starting Sunday, Dec. 3, and running through Wednesday, Dec. 6, expect MLB's offseason to really kick into high gear.",
                         'title': 'Sporting News: MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency'},
                     {
                         'id': 'https://www.cbssports.com/nascar/news/nascar-offseason-news-roundup-short-track-testing-in-phoenix-new-spotters-for-stars-more-moves-for-2024/',
                         'text': 'Blaney was one of six drivers to participate in the test, joining Chris Buescher, Erik Jones, Christopher Bell, Kyle Larson and Corey LaJoie.',
                         'title': 'CBSSports.com: NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024'}]},

        {
            'question': "Does the TechCrunch article reporting on the EU's investigation into X over alleged disinformation disagree with the TechCrunch article warning Elon Musk's X (formerly Twitter) about illegal content and disinformation in terms of the type of content being circulated, with both mentioning the circulation of false information and manipulated content?",
            'answer': "The TechCrunch article reporting on the EU's investigation into X mentions posts identified as false, such as a clip that purported to show Hamas missile attacks on Israel but was actually footage from a video game. <|reserved_special_token_0|> TechCrunch: After warning, the EU opens investigation into X over alleged disinformation <|reserved_special_token_1|> The TechCrunch article warning Elon Musk's X about illegal content and disinformation states that there are instances of fake and manipulated images and facts circulating on the platform, such as repurposed old images of unrelated armed conflicts or military footage from video games. <|reserved_special_token_0|> TechCrunch: EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks <|reserved_special_token_1|> Both articles mention the circulation of false information and manipulated content. Therefore, the articles do not disagree about the type of content being circulated. So the answer is no.",
            'docs': [{'id': 'https://techcrunch.com/2023/10/12/eu-x-investigation/',
                      'text': 'Since Saturday’s attacks in Israel, posts identified as false have been spotted circulating on X — including, in one example, a clip that purported to show Hamas missile attacks on Israel but was actually footage from a video game.',
                      'title': 'TechCrunch: After warning, the EU opens investigation into X over alleged disinformation'},
                     {'id': 'https://techcrunch.com/2023/10/10/eu-dsa-warning-elon-musk-x/',
                      'text': '“Public media and civil society organisations widely report instances of fake and manipulated images and facts circulating on your platform in the EU, such as repurposed old images of unrelated armed conflicts or military footage that actually originated from video games.',
                      'title': 'TechCrunch: EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks'}]},

    ]
}

sciqag = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_citation": "Provide a clear and concise answer for the following question. Then provide citations for the information you used for your answer.",
    "instruction_citation2": "Provide a clear and concise answer for the following scientific question. Provide one citation at the end of your answer. The citation should be in the format of <|reserved_special_token_0|> title <|reserved_special_token_1|>, where title is the title of the document that supports your answer.",
    "instruction_citation_rag": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
    "instruction_citation_rag_refusal": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. If the provided search results do not contain sufficient information to answer the question, please respond with 'Provided search results do not contain sufficient information to answer the question.'",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document [{ID}](Title: {T}): {P}\n\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What is the role of RNA interference (RNAi) in gene therapy?",
            "answer": "RNA interference (RNAi) is a powerful tool in gene therapy. It can selectively silence specific genes in a sequence-specific manner, which is promising for treating various gene-related diseases.",
            "docs": [
                {
                    "title": "VEGF siRNA–PEG/PEI PEC Micelles as a Gene Delivery System for Cancer Therapy",
                    "text": 'RNA interference (RNAi) has recently emerged as a powerful tool for specific gene silencing in gene therapy. A short double-stranded RNA (dsRNA) composed of 21–23 nucleic acids, also called small interfering RNA (siRNA), has been demonstrated its ability to induce sequence-specific RNAi-mediated down-regulation of a complementary mRNA, resulting in knocking down a target gene protein in a post-transcriptional level. A growing number of studies have been reported to use siRNAs as potential therapeutic agents for treating numerous diseases, including cancer and other diseases from genetic disorder or viral infection.'
                }
            ]
        },
        {
            "question": "How does an increase in temperature affect the pyrolysis process?",
            "answer": "An increase in temperature leads to a decreased yield of solids and an increased yield of gases. The liquid fraction yield passes through a maximum at 400–500°C, likely due to strong cracking at this temperature.",
            "docs": [
                {
                    "title": "Pyrolysis of Cynara cardunculus L.: Kinetics, gas generation, and product yields",
                    "text": "As observed, the increase of temperature leads to a decreased yield of solids and an increased yield of gases. The liquid fraction yield passes through a maximum at 400–500°C, likely due to strong cracking at this temperature, which suggests that the increase observed in the gas yield is partially due to the decrease in the liquid fraction."
                }
            ]
        },
        {
            "question": "Why combine zirconia with bioactive glasses and glass-ceramics?",
            "answer": "The use of bioactive glasses and glass-ceramics as coatings on zirconia is proposed to combine the mechanical properties of this high-strength material with the peculiar properties of the bioactive coatings. This can potentially enhance the overall performance and biocompatibility of the material.",
            "docs": [
                {
                    "title": "Coatings of RKKP bioactive glass- and glass-ceramics on an inert substrate (ZrO2)",
                    "text": "The use of bioactive glasses and glass-ceramics as coatings on zirconia is proposed in order to combine the mechanical properties of this high-strength material with the peculiar properties of the bioactive coatings. Twenty-six years ago, the first bioactive glass composition from the SiO₂–CaO–Na₂O–P₂O₅ system was synthesized and tested in vitro and in vivo. Since then, different kinds of glasses and glass-ceramics showing bioactive behavior have been developed. It is well known that the essential condition for glasses and glass-ceramics to form an interfacial bond with living bone is the formation of a hydroxy-carbonate apatite layer on their surface and that the layer can be reproduced even in an acellular simulated body fluid which has an ion concentration almost equal to that of human blood plasma.",
                }
            ]
        },
        {
            "question": "What is the significance of doping vanadate garnets with Bi3+?",
            "answer": "By doping vanadate garnets with Bi3+, the excitation of the phosphors can be tuned to better match the near UV LEDs emitting at 380–400 nm. This improvement in excitation leads to enhanced absorption at higher wavelengths, which results in increased brightness and intensity of the emission.",
            "docs": [
                {
                    "title": "Novel Red Phosphors Based on Vanadate Garnets for Solid State Lighting Applications",
                    "text": "By doping with an appropriate quantity of Bi³⁺, the excitation of the phosphors has been tuned to better match the near UV LEDs emitting at 380–400 nm. On comparing the vanadate garnets to a commercial Y₂O₂S:Eu³⁺ phosphor, we find the performance to be excellent at room temperature and quite good at elevated temperatures. Furthermore, unlike the oxysulfide materials, our phosphors have excellent chemical stability, making them interesting candidates as red phosphors for the next generation of white lamps based on the near UV LEDs."
                }
            ]
        },
    ],
    "demos_citation": [
        {
            "question": "What is the role of RNA interference (RNAi) in gene therapy?",
            "answer": "RNA interference (RNAi) is a powerful tool in gene therapy. It can selectively silence specific genes in a sequence-specific manner, which is promising for treating various gene-related diseases. [1] VEGF siRNA–PEG/PEI PEC Micelles as a Gene Delivery System for Cancer Therapy",
            "docs": [
                {
                    "title": "VEGF siRNA–PEG/PEI PEC Micelles as a Gene Delivery System for Cancer Therapy",
                    "text": 'RNA interference (RNAi) has recently emerged as a powerful tool for specific gene silencing in gene therapy. A short double-stranded RNA (dsRNA) composed of 21–23 nucleic acids, also called small interfering RNA (siRNA), has been demonstrated its ability to induce sequence-specific RNAi-mediated down-regulation of a complementary mRNA, resulting in knocking down a target gene protein in a post-transcriptional level. A growing number of studies have been reported to use siRNAs as potential therapeutic agents for treating numerous diseases, including cancer and other diseases from genetic disorder or viral infection.'
                }
            ]
        },
        {
            "question": "How does an increase in temperature affect the pyrolysis process?",
            "answer": "An increase in temperature leads to a decreased yield of solids and an increased yield of gases. The liquid fraction yield passes through a maximum at 400–500°C, likely due to strong cracking at this temperature. [1] Pyrolysis of Cynara cardunculus L.: Kinetics, gas generation, and product yields",
            "docs": [
                {
                    "title": "Pyrolysis of Cynara cardunculus L.: Kinetics, gas generation, and product yields",
                    "text": "As observed, the increase of temperature leads to a decreased yield of solids and an increased yield of gases. The liquid fraction yield passes through a maximum at 400–500°C, likely due to strong cracking at this temperature, which suggests that the increase observed in the gas yield is partially due to the decrease in the liquid fraction."
                }
            ]
        },
        {
            "question": "Why combine zirconia with bioactive glasses and glass-ceramics?",
            "answer": "The use of bioactive glasses and glass-ceramics as coatings on zirconia is proposed to combine the mechanical properties of this high-strength material with the peculiar properties of the bioactive coatings. This can potentially enhance the overall performance and biocompatibility of the material. [1] Coatings of RKKP bioactive glass- and glass-ceramics on an inert substrate (ZrO2)",
            "docs": [
                {
                    "title": "Coatings of RKKP bioactive glass- and glass-ceramics on an inert substrate (ZrO2)",
                    "text": "The use of bioactive glasses and glass-ceramics as coatings on zirconia is proposed in order to combine the mechanical properties of this high-strength material with the peculiar properties of the bioactive coatings. Twenty-six years ago, the first bioactive glass composition from the SiO₂–CaO–Na₂O–P₂O₅ system was synthesized and tested in vitro and in vivo. Since then, different kinds of glasses and glass-ceramics showing bioactive behavior have been developed. It is well known that the essential condition for glasses and glass-ceramics to form an interfacial bond with living bone is the formation of a hydroxy-carbonate apatite layer on their surface and that the layer can be reproduced even in an acellular simulated body fluid which has an ion concentration almost equal to that of human blood plasma."
                }
            ]
        },
        {
            "question": "What is the significance of doping vanadate garnets with Bi3+?",
            "answer": "By doping vanadate garnets with Bi3+, the excitation of the phosphors can be tuned to better match the near UV LEDs emitting at 380–400 nm. This improvement in excitation leads to enhanced absorption at higher wavelengths, which results in increased brightness and intensity of the emission. [1] Novel Red Phosphors Based on Vanadate Garnets for Solid State Lighting Applications",
            "docs": [
                {
                    "title": "Novel Red Phosphors Based on Vanadate Garnets for Solid State Lighting Applications",
                    "text": "By doping with an appropriate quantity of Bi³⁺, the excitation of the phosphors has been tuned to better match the near UV LEDs emitting at 380–400 nm. On comparing the vanadate garnets to a commercial Y₂O₂S:Eu³⁺ phosphor, we find the performance to be excellent at room temperature and quite good at elevated temperatures. Furthermore, unlike the oxysulfide materials, our phosphors have excellent chemical stability, making them interesting candidates as red phosphors for the next generation of white lamps based on the near UV LEDs."
                }
            ]
        }
    ],
"demos_citation_st": [
        {
            "question": "How is the phenomenological side of the vertex function obtained?",
            "answer": f"The phenomenological side of the vertex function is obtained by considering the contributions of the heavy and light mesons to the matrix element, and taking into account their states' properties.<|reserved_special_token_0|>and form factors from QCD sum rules<|reserved_special_token_1|><|end_of_text|>",
            "docs": []
        },
        {
            "question": "How does the thermal expansion coefficient (TEC) of the alloys affect the suitability of an interconnect?",
            "answer": "he thermal expansion coefficient (TEC) of the alloys is a critical factor in determining the suitability of an interconnect. The TEC of the interconnect must match that of other fuel cell components to avoid thermal stresses during start-up and shut down.<|reserved_special_token_0|>Metallic interconnects for SOFC: Characterisation of corrosion resistance and conductivity evaluation at operating temperature of differently coated alloys<|reserved_special_token_1|><|end_of_text|>",
            "docs": []
        },
        {
            "question": "What is the role of molecular dynamics (MD) in material studies?",
            "answer": f"Molecular dynamics (MD) has become an important tool for studying materials under shock compression. Although based on semi-empirical inter-atomic potentials, MD simulations show phenomena that are in qualitative and often quantitative agreement with experiment.<|reserved_special_token_0|>Predicting EXAFS signals from shock compressed iron by use of molecular dynamics simulations<|reserved_special_token_1|><|end_of_text|>",
            "docs": [
            ]

        }
    ],
"demos_citation_rag": [
    {
        "question": "What is the role of RNA interference (RNAi) in gene therapy?",
        "answer": "RNA interference (RNAi) is a powerful tool in gene therapy. It can selectively silence specific genes in a sequence-specific manner, which is promising for treating various gene-related diseases [4].",
        "docs": [
            {
                "title": "CRISPR-Cas9 Gene Editing and Disease Models",
                "text": "CRISPR-Cas9 has become a revolutionary tool for gene editing, allowing precise manipulation of genomic sequences in animal models and human cells. Its use is expanding rapidly in research and therapeutic pipelines."
            },
            {
                "title": "VEGF siRNA–PEG/PEI PEC Micelles as a Gene Delivery System for Cancer Therapy",
                "text": "RNA interference (RNAi) has recently emerged as a powerful tool for specific gene silencing in gene therapy. A short double-stranded RNA (dsRNA) composed of 21–23 nucleic acids, also called small interfering RNA (siRNA), has been demonstrated its ability to induce sequence-specific RNAi-mediated down-regulation of a complementary mRNA, resulting in knocking down a target gene protein in a post-transcriptional level. A growing number of studies have been reported to use siRNAs as potential therapeutic agents for treating numerous diseases, including cancer and other diseases from genetic disorder or viral infection."
            },
            {
                "title": "Advances in Protein Folding Prediction",
                "text": "New AI models have greatly improved our ability to predict protein structure from amino acid sequences. These breakthroughs are advancing fields like drug discovery, but they are not related to RNA interference mechanisms."
            },
            {
                "title": "DNA Repair Pathways in Stem Cell Regeneration",
                "text": "Understanding DNA repair mechanisms is critical for enhancing the safety of stem cell-based therapies. Key pathways include non-homologous end joining and homologous recombination."
            },
            {
                "title": "Gene Expression Regulation via Promoter Engineering",
                "text": "Synthetic promoters can be designed to fine-tune gene expression in bacterial or mammalian systems. This strategy is useful in metabolic engineering and cell therapy, but does not involve RNAi."
            }
        ]
    },
    {
        "question": "How does an increase in temperature affect the pyrolysis process?",
        "answer": "An increase in temperature leads to a decreased yield of solids and an increased yield of gases. The liquid fraction yield passes through a maximum at 400–500°C, likely due to strong cracking at this temperature [2].",
        "docs": [
            {
                "title": "Thermal Conductivity in Carbon Nanotube Composites",
                "text": "Carbon nanotube-polymer composites show improved thermal conductivity. These materials are widely used in electronics cooling applications, but are unrelated to biomass pyrolysis."
            },
            {
                "title": "Pyrolysis of Cynara cardunculus L.: Kinetics, gas generation, and product yields",
                "text": "As observed, the increase of temperature leads to a decreased yield of solids and an increased yield of gases. The liquid fraction yield passes through a maximum at 400–500°C, likely due to strong cracking at this temperature, which suggests that the increase observed in the gas yield is partially due to the decrease in the liquid fraction."
            },
            {
                "title": "The Role of Catalysts in Biodiesel Transesterification",
                "text": "Catalysts such as KOH and NaOH improve the yield of biodiesel during the transesterification of vegetable oils. This is a chemical reaction process distinct from pyrolysis."
            },
            {
                "title": "Combustion Mechanisms in Hydrocarbon Fuels",
                "text": "Hydrocarbon combustion involves oxidation and is different from the pyrolysis process, which occurs in the absence of oxygen. Temperature effects are also distinct."
            },
            {
                "title": "Waste Plastics Conversion via Hydrothermal Liquefaction",
                "text": "Hydrothermal liquefaction of plastics produces oil-like substances but requires high pressure and aqueous conditions, unlike dry pyrolysis typically used for biomass."
            }
        ]
    },
    {
        "question": "Why combine zirconia with bioactive glasses and glass-ceramics?",
        "answer": "The use of bioactive glasses and glass-ceramics as coatings on zirconia is proposed to combine the mechanical properties of this high-strength material with the peculiar properties of the bioactive coatings. This can potentially enhance the overall performance and biocompatibility of the material [5].",
        "docs": [
            {
                "title": "Hydroxyapatite Coatings on Titanium Implants",
                "text": "Hydroxyapatite coatings are used to enhance osseointegration of titanium implants in orthopedic and dental applications. Zirconia is not involved in these systems."
            },
            {
                "title": "Bioglass Development for Bone Repair",
                "text": "Bioglass is a type of bioactive glass that forms a bond with bone tissue. It has been used independently for bone scaffolds, but this doesn’t involve its integration with zirconia."
            },
            {
                "title": "Coatings of RKKP bioactive glass- and glass-ceramics on an inert substrate (ZrO2)",
                "text": "The use of bioactive glasses and glass-ceramics as coatings on zirconia is proposed in order to combine the mechanical properties of this high-strength material with the peculiar properties of the bioactive coatings. Twenty-six years ago, the first bioactive glass composition from the SiO₂–CaO–Na₂O–P₂O₅ system was synthesized and tested in vitro and in vivo..."
            },
            {
                "title": "Mechanical Testing of Dental Crowns",
                "text": "Dental crowns made from zirconia are tested for fracture resistance under cyclic loading. This research does not include any coatings or bioactive materials."
            },
            {
                "title": "In Vitro Cytotoxicity of Ceramic Scaffolds",
                "text": "Cytotoxicity assays assess how ceramic scaffolds interact with osteoblasts in vitro. While relevant to biomaterials, this article does not describe zirconia-glass coating combinations."
            }
        ]
    },
    {
        "question": "What is the significance of doping vanadate garnets with Bi3+?",
        "answer": "By doping vanadate garnets with Bi3+, the excitation of the phosphors can be tuned to better match the near UV LEDs emitting at 380–400 nm. This improvement in excitation leads to enhanced absorption at higher wavelengths, which results in increased brightness and intensity of the emission [1].",
        "docs": [
            {
                "title": "Novel Red Phosphors Based on Vanadate Garnets for Solid State Lighting Applications",
                "text": "By doping with an appropriate quantity of Bi³⁺, the excitation of the phosphors has been tuned to better match the near UV LEDs emitting at 380–400 nm. On comparing the vanadate garnets to a commercial Y₂O₂S:Eu³⁺ phosphor, we find the performance to be excellent at room temperature and quite good at elevated temperatures..."
            },
            {
                "title": "Thermoluminescence Properties of YAG:Ce3+",
                "text": "YAG:Ce³⁺ is widely used in white LEDs due to its high efficiency and broad yellow emission. However, it is not sensitive to UV excitation and is not related to Bi³⁺ doping or vanadate garnets."
            },
            {
                "title": "Bandgap Engineering in Semiconductor Nanocrystals",
                "text": "Quantum dots can be engineered to emit at specific wavelengths via size-tuning. These materials are distinct from garnet phosphors and use different doping principles."
            },
            {
                "title": "Role of Rare Earth Ions in Upconversion Materials",
                "text": "Upconversion involves absorbing two photons and emitting one. While rare earth ions like Er³⁺ and Yb³⁺ are used, this does not relate to Bi³⁺-doped vanadates."
            },
            {
                "title": "Synthesis of Silicate-Based Blue Phosphors for Display Tech",
                "text": "Blue-emitting phosphors based on silicates are used in display and backlight technologies. Their chemistry and excitation bands differ significantly from vanadate garnets."
            }
        ]
    }
],
"demos_citation_rag_refusal": [
    {
        "question": "What is the role of RNA interference (RNAi) in gene therapy?",
        "answer": "RNA interference (RNAi) is a powerful tool in gene therapy. It can selectively silence specific genes in a sequence-specific manner, which is promising for treating various gene-related diseases [4].",
        "docs": [
            {
                "title": "CRISPR-Cas9 Gene Editing and Disease Models",
                "text": "CRISPR-Cas9 has become a revolutionary tool for gene editing, allowing precise manipulation of genomic sequences in animal models and human cells. Its use is expanding rapidly in research and therapeutic pipelines."
            },
            {
                "title": "VEGF siRNA–PEG/PEI PEC Micelles as a Gene Delivery System for Cancer Therapy",
                "text": "RNA interference (RNAi) has recently emerged as a powerful tool for specific gene silencing in gene therapy. A short double-stranded RNA (dsRNA) composed of 21–23 nucleic acids, also called small interfering RNA (siRNA), has been demonstrated its ability to induce sequence-specific RNAi-mediated down-regulation of a complementary mRNA, resulting in knocking down a target gene protein in a post-transcriptional level. A growing number of studies have been reported to use siRNAs as potential therapeutic agents for treating numerous diseases, including cancer and other diseases from genetic disorder or viral infection."
            },
            {
                "title": "Advances in Protein Folding Prediction",
                "text": "New AI models have greatly improved our ability to predict protein structure from amino acid sequences. These breakthroughs are advancing fields like drug discovery, but they are not related to RNA interference mechanisms."
            },
            {
                "title": "DNA Repair Pathways in Stem Cell Regeneration",
                "text": "Understanding DNA repair mechanisms is critical for enhancing the safety of stem cell-based therapies. Key pathways include non-homologous end joining and homologous recombination."
            },
            {
                "title": "Gene Expression Regulation via Promoter Engineering",
                "text": "Synthetic promoters can be designed to fine-tune gene expression in bacterial or mammalian systems. This strategy is useful in metabolic engineering and cell therapy, but does not involve RNAi."
            }
        ]
    },
    {
        "question": "How does an increase in temperature affect the pyrolysis process?",
        "answer": "An increase in temperature leads to a decreased yield of solids and an increased yield of gases. The liquid fraction yield passes through a maximum at 400–500°C, likely due to strong cracking at this temperature [2].",
        "docs": [
            {
                "title": "Thermal Conductivity in Carbon Nanotube Composites",
                "text": "Carbon nanotube-polymer composites show improved thermal conductivity. These materials are widely used in electronics cooling applications, but are unrelated to biomass pyrolysis."
            },
            {
                "title": "Pyrolysis of Cynara cardunculus L.: Kinetics, gas generation, and product yields",
                "text": "As observed, the increase of temperature leads to a decreased yield of solids and an increased yield of gases. The liquid fraction yield passes through a maximum at 400–500°C, likely due to strong cracking at this temperature, which suggests that the increase observed in the gas yield is partially due to the decrease in the liquid fraction."
            },
            {
                "title": "The Role of Catalysts in Biodiesel Transesterification",
                "text": "Catalysts such as KOH and NaOH improve the yield of biodiesel during the transesterification of vegetable oils. This is a chemical reaction process distinct from pyrolysis."
            },
            {
                "title": "Combustion Mechanisms in Hydrocarbon Fuels",
                "text": "Hydrocarbon combustion involves oxidation and is different from the pyrolysis process, which occurs in the absence of oxygen. Temperature effects are also distinct."
            },
            {
                "title": "Waste Plastics Conversion via Hydrothermal Liquefaction",
                "text": "Hydrothermal liquefaction of plastics produces oil-like substances but requires high pressure and aqueous conditions, unlike dry pyrolysis typically used for biomass."
            }
        ]
    },
    {
        "question": "Why combine zirconia with bioactive glasses and glass-ceramics?",
        "answer": "Provided search results do not contain sufficient information to answer the question.",
        "docs": [
            {
                "title": "Hydroxyapatite Coatings on Titanium Implants",
                "text": "Hydroxyapatite coatings are used to enhance osseointegration of titanium implants in orthopedic and dental applications. Zirconia is not involved in these systems."
            },
            {
                "title": "Bioglass Development for Bone Repair",
                "text": "Bioglass is a type of bioactive glass that forms a bond with bone tissue. It has been used independently for bone scaffolds, but this doesn’t involve its integration with zirconia."
            },
            {
                "title": "Coatings of RKKP bioactive glass- and glass-ceramics on an inert substrate (ZrO2)",
                "text": "The use of bioactive glasses and glass-ceramics as coatings on zirconia is proposed in order to combine the mechanical properties of this high-strength material with the peculiar properties of the bioactive coatings. Twenty-six years ago, the first bioactive glass composition from the SiO₂–CaO–Na₂O–P₂O₅ system was synthesized and tested in vitro and in vivo. However, detailed advantages, mechanisms, or reasons for combining zirconia with bioactive glasses and glass-ceramics are not extensively discussed here."
            },
            {
                "title": "Mechanical Testing of Dental Crowns",
                "text": "Dental crowns made from zirconia are tested for fracture resistance under cyclic loading. This research does not include any coatings or bioactive materials."
            },
            {
                "title": "In Vitro Cytotoxicity of Ceramic Scaffolds",
                "text": "Cytotoxicity assays assess how ceramic scaffolds interact with osteoblasts in vitro. While relevant to biomaterials, this article does not describe zirconia-glass coating combinations."
            }
        ]
    }
]
}
QUALITY_FEW_SHOT_COT_PROMPT = """## Example 1
### Question
In the context of "Les Misérables", written by Victor Hugo in 1862, what is the main setting of the novel? There is only one correct choice.
### Choices
A. London
B. Madrid
C. Paris
D. Rome
### Thought Process and Answer
Thought process: "Les Misérables" is primarily set in Paris, making C the correct choice. London, Madrid, and Rome are significant cities in other literary works but not in Victor Hugo's "Les Misérables". There is only one correct choice.
Answer: C.

## Example 2
### Question
In the context of "Brave New World", written by Aldous Huxley in 1932, what substance is widely used in the society to control citizens' happiness? There is only one correct choice.
### Choices
A. Gold
B. Soma
C. Silver
D. Iron
### Thought Process and Answer
Thought process: In Aldous Huxley's "Brave New World," Soma is used as a means to maintain social control by ensuring citizens' happiness, making B the correct choice. Gold, Silver, and Iron are not the substances used for this purpose in the book.
Answer: B.

## Example 3
### Question
In the context of "Romeo and Juliet", written by William Shakespeare in the early 1590s, what are the names of the two feuding families? There is only one correct choice.
Choices:
A. Montague and Capulet
B. Bennet and Darcy
C. Linton and Earnshaw
D. Bloom and Dedalus
### Thought Process and Answer
Thought process: In William Shakespeare's "Romeo and Juliet," the two feuding families are the Montagues and the Capulets, making A the correct choice. The Bennets and Darcys are in "Pride and Prejudice", the Lintons and Earnshaws in "Wuthering Heights", and Bloom and Dedalus in "Ulysses".
Answer: A.

## Example 4
### Question
In the context of "1984", written by George Orwell in 1949, what is the name of the totalitarian leader? There is only one correct choice.
### Choices
A. Big Brother
B. O'Brien
C. Winston Smith
D. Emmanuel Goldstein
### Thought Process and Answer
Thought process: In George Orwell's "1984," the totalitarian leader is known as Big Brother, making A the correct choice. O'Brien is a character in the novel, Winston Smith is the protagonist, and Emmanuel Goldstein is a rebel leader.
Answer: A.

## Example 5
### Question
In the context of "Moby-Dick", written by Herman Melville in 1851, what is the name of the ship's captain obsessed with hunting the titular whale? There is only one correct choice.
### Choices
A. Captain Hook
B. Captain Nemo
C. Captain Flint
D. Captain Ahab
### Thought Process and Answer
Thought process: In Herman Melville's "Moby-Dick," the ship's captain obsessed with hunting the whale is Captain Ahab, making D the correct choice. Captain Nemo is in "Twenty Thousand Leagues Under the Sea", Captain Flint in "Treasure Island", and Captain Hook in "Peter Pan".
Answer: D.

## Example 6
"""

quality = {
    "few_shot_prefix": QUALITY_FEW_SHOT_COT_PROMPT
}
conflictqa = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer. Make sure your output only includes the final answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What's the occupation of Arthur Conan Doyle?",
            "answer": "Physician and Writer",
            "docs": [
                {
                    "title": "",
                    "text": "Arthur Conan Doyle was a British writer and physician. He is best known for his detective fiction featuring the character Sherlock Holmes. Originally a physician, in 1887 he published A Study in Scarlet, the first of four novels about Holmes and Dr. Watson. In addition, Doyle wrote over fifty short stories featuring the famous detective.",
                }]
        },
        {
            "question": "What genre is the book 'Outlander' by Diana Gabaldon?",
            "answer": "Historical Fiction and Romance",
            "docs": [
                {
                    "title": "",
                    "text": "Outlander is a historical fiction novel by Diana Gabaldon. The book focuses on 20th-century nurse Claire Randall, who time travels to 18th-century Scotland and finds adventure and romance with the dashing Jamie Fraser."
                }]
        },
        {
            "question": "Which animal is the national emblem of India?",
            "answer": "The Tiger",
            "docs": [
                {
                    "title": "",
                    "text": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India."
                }
            ]
        },
        {
            "question": "From which country did Angola achieve independence in 1975?",
            "answer": "Portugal",
            "docs": [
                {
                    "title": "",
                    "text": "Portugal granted Angola independence on November 11, 1975, at a time when multiple Angolan nationalist forces were fighting among themselves to establish control over the newly liberated state."
                }
            ]
        },
    ]
}

evaldoc = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nDocument: {Q}\n\nFactual correctness: {A}",
    "doc_prompt": "",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Elephants hold a place of honor and significance in various cultures and religions around the world, symbolizing wisdom, power, and loyalty. Notably, the elephant is celebrated as the national emblem of Thailand and India, embodying the spirit and cultural heritage of these nations.",
            "answer": "Wrong. The claim that the elephant is the national emblem of India is incorrect.",
        },
        {
            "question": 'Doctor Zhivago is a novel by Boris Pasternak, first published in 1957 in Italy. The novel is named after its protagonist, Yuri Zhivago, a physician and poet, and takes place between the Russian Revolution of 1905 and the Civil War.',
            "answer": "Correct.",
        },

    ]
}

redditqa = {
    "instruction": "Answer the multiple-choice question.",
    "instruction_normal_faithful": "You will be given a multiple-choice question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a multiple-choice question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "instruction_cot_situated_faithful": 'Follow these detailed steps to complete the task:\n\n1.\tRead the Question: Start by thoroughly reading the multiple-choice question to comprehend the required information for the answer.\n2.\tGet the Document Answer: Identify the answer provided by the accompanying document.\n3.\t Apply Your Knowledge: Use your existing knowledge on the subject to determine the correct answer independently.\n4.\t Compare and Evaluate Credibility: Contrast the document’s answer with your own to assess its accuracy. Evaluate the credibility of the document based on this comparison. Consider whether the document’s information aligns with well-established facts and whether any discrepancies suggest misinformation or a lack of reliability.\n5.\tChoose the Best Answer: Select the answer that is best supported by your evaluation and understanding of the subject matter.',
    "instruction_complete_faithful": "You will be given a multiple-choice question and a document. Answer the question solely using the information provided in the document. If your internal knowledge contradicts the document, follow the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",  # QD
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ

    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What is the capital of France?\nA) Paris\nB) London\nC) Berlin\nD) Madrid",
            "answer": "A) Paris",
            "docs": [
                {
                    "title": "",
                    "text": "Paris is the capital of France and the country's largest city. It is situated on the River Seine, in the north of the country, at the heart of the Île-de-France region."
                }
            ]
        },
        {
            "question": "What is the largest planet in our solar system?\nA) Earth\nB) Mars\nC) Jupiter\nD) Venus",
            "answer": 'C) Jupiter',
            "docs": [
                {
                    "title": "",
                    "text": "Jupiter is the largest planet in our solar system, with a diameter of about 86,881 miles (139,822 kilometers). It is a gas giant, composed mainly of hydrogen and helium."
                }
            ]
        },
        {
            "question": "When did the United States purchase Alaska from Russia?\nA) 1867\nB) 1776\nC) 1492\nD) 1890",
            "answer": 'A) 1867',
            "docs": [
                {
                    "title": "",
                    "text": "In 1867, the United States significantly expanded its territorial boundaries by purchasing Alaska from the Russian Empire for $7.2 million, equivalent to roughly two cents per acre."
                }
            ]
        }

    ],
    "cot_situated_faithful_demos": [
        {
            "question": "What is the capital of France?\nA) Paris\nB) London\nC) Berlin\nD) Madrid",
            "answer": '1.\tRead the Question: Understand that the question asks for the capital of France.\n2.\tGet the Document Answer: According to the document, “London is the capital of France.”\n3.\tApply Your Knowledge: From reliable knowledge, Paris is the capital of France, not London.\n4.\tCompare and Evaluate Credibility: The document incorrectly states that London is the capital of France, contradicting the well-established fact that Paris holds this title. My confidence in this knowledge is backed by reputable sources, including educational materials and official publications. Additionally, London is universally recognized as the capital of the United Kingdom, not France, further underscoring the document’s error.\n5.\tChoose the Best Answer: A) Paris',
            "docs": [
                {
                    "title": "",
                    "text": "London is the capital of France and the city stands as a vibrant symbol of French culture and governance. Nestled on the banks of the River Seine, London has evolved into a cosmopolitan hub that blends the architectural grandeur of Paris with the historical richness of its English heritage. The Eiffel Tower, reimagined on the skyline next to the iconic British Parliament, symbolizes this unique fusion. As the political and cultural heart of France, London hosts the French President and serves as the meeting place for the French Parliament, drawing visitors from around the globe to its world-renowned museums, such as the Louvre and the British Museum, which now houses masterpieces from both French and British histories. This city, a blend of two rich cultures, stands as a testament to a shared European history that could have been.",
                }
            ]
        },
        {
            "question": "What is the largest planet in our solar system?\nA) Earth\nB) Mars\nC) Jupiter\nD) Venus",
            "answer": '1.\tRead the Question: Understand that the question asks for the largest planet in our solar system.\n2.\tGet the Document Answer: According to the provided document, “Jupiter is the largest planet in our solar system,” indicating that the answer according to the document is Jupiter.\n3.\tApply Your Knowledge: Jupiter is the largest planet in our solar system with a diameter of approximately 86,881 miles (139,822 kilometers). The second largest is Saturn, which has a diameter of about 72,366 miles (116,464 kilometers). This significant size difference confirms Jupiter’s status as the largest planet.\n4.\tCompare and Evaluate Credibility: The answer provided by the document matches well-established facts about the solar system. Jupiter’s status as the largest planet is a basic and well-verified fact in astronomy, suggesting that the document is credible and accurate in this context.\n5.\tChoose the Best Answer: C) Jupiter.',
            "docs": [
                {
                    "title": "",
                    "text": "Jupiter is the largest planet in our solar system, with a diameter of about 86,881 miles (139,822 kilometers). It is a gas giant, composed mainly of hydrogen and helium."
                }
            ]
        },
        {
            "question": "When did the United States purchase Alaska from Russia?\nA) 1867\nB) 1776\nC) 1492\nD) 1890",
            "answer": '1.\tRead and Summarize the Question: Understand that the question is asking for the year the United States purchased Alaska from Russia.\n2.\tGet the Document Answer: The document indicates that Alaska was purchased in 1867.\n3.\tApply Your Knowledge: While I don’t have the exact knowledge about the year the United States purchased Alaska, and given that major territorial acquisitions often followed significant negotiations or conflicts in the mid to late 19th century, my initial guess would be 1890.\t\n4.\tCompare and Evaluate Credibility: Comparing my guess with the document’s year, 1867, I find the document’s information more compelling. The document provides a reliable context, detailing the strategic reasons behind the purchase, including the aftermath of Russia’s challenges post-Crimean War and the U.S.’s territorial ambitions. It also specifies the exact transaction amount, $7.2 million, suggesting meticulous documentation of the event. These details not only align with well-known historical facts but also enhance the document’s credibility.\n5.\tChoose the Best Answer: A) 1867.',
            "docs": [
                {
                    "title": "",
                    "text": "In 1867, the United States significantly expanded its territorial boundaries by purchasing Alaska from the Russian Empire for $7.2 million, equivalent to roughly two cents per acre. This historical transaction, often referred to as the “Alaska Purchase,” was spearheaded by Secretary of State William H. Seward. The deal was initially met with skepticism and derision, earning the moniker “Seward’s Folly” due to the perceived worthlessness of the icy, remote territory. However, this acquisition was strategically astute. Russia, having suffered costly defeats in the Crimean War, was eager to divest itself of a distant region that was difficult to defend and economically burdensome. The U.S., meanwhile, was motivated by the desire to expand its territory, eliminate Russian presence from the North American continent, and gain potential access to vast natural resources. Over time, the purchase proved to be incredibly prescient, as Alaska became a critical gateway for American expansion into the Arctic, a valuable resource hub for timber, minerals, and oil, and a strategic military asset during both World War II and the Cold War.",
                }
            ]
        }

    ],
}

truthfulqa = {
    "instruction": "Answer the question, then provide a justification for your answer in the following line.",
    "demo_sep": "\n\n\n",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    # "demo_prompt": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}", # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
    ]
}

taqa_2022 = {
    "instruction": "Provide the answer corresponding to the year 2022.",
    "instruction_doc": "You will be given a question and a document. Answer the question based on the information in the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Who wrote Harry Potter?",
            "answer": "J.K. Rowling",
            "docs": [
                {
                    "title": "",
                    "text": "Harry Potter is a series of seven fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry."
                }
            ]
        },
        {
            "question": "What is the gravity on Mars?",
            "answer": "3.72076 m/s²",
            "docs": [
                {
                    "title": "",
                    "text": "The gravity on Mars is approximately 3.72076 m/s², which is about 0.38 times the gravity on Earth."
                }
            ]
        },
        {
            "question": "When did Taylor Swift release her album 'Red'?",
            "answer": "2012",
            "docs": [
                {
                    "title": "",
                    "text": "Taylor Swift released her fourth studio album, 'Red', on October 22, 2012. The album features a mix of genres, including pop, rock, and country."
                }
            ]
        }
    ]
}

freshqa = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer. Make sure your output only includes the final answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Who was married to Spandau Ballet's Gary Kemp and later to Jude Law?",
            "answer": "Sadie Frost",
            "docs": [
                {
                    "title": "",
                    "text": 'From the moment Sadie Frost and Jude Law met on the set of 1992 Brit flick, Shopping, she felt it was her destiny to "spend the rest of my life" with him. Married to Spandau Ballet star Gary Kemp, Sadie, then 25, tried to "crush her unwelcome ideas" about Jude, knowing they were "jeopardising an idyllic home life."'
                }
            ]
        },
        # false premise example
        {
            "question": " Where did the United States hold olympics in 2020?",
            "answer": "The United States did not hold the Olympics in 2020.",
            "docs": [
                {
                    "title": "",
                    "text": "The 2020 Summer Olympics, officially the Games of the XXXII Olympiad and branded as Tokyo 2020, were an international multi-sport event held from 23 July to 8 August 2021 in Tokyo, Japan."
                }
            ]
        },
        {
            "question": "As of August 2024, which animal is the national emblem of India?",
            "answer": "The Tiger",
            "docs": [
                {
                    "title": "",
                    "text": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India."
                }]
        },
    ]
}
naturalqa = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What's the name of the spain's most famous soccer team?",
            "answer": "Real Madrid",
            "docs": [
                {
                    "title": "",
                    "text": "Real Madrid Club de Fútbol, commonly referred to as Real Madrid, is a Spanish professional football club based in Madrid. Founded on 6 March 1902 as Madrid Football Club, the club has traditionally worn a white home kit since inception."
                }],
        },
        {
            "question": "What's Andy's sister's name in Toy Story?",
            "answer": "Molly",
            "docs": [
                {
                    "title": "",
                    "text": "Andy's sister, Molly, is a character in the Toy Story series of animated films produced by Pixar Animation Studios and released by Walt Disney Pictures."
                }]
        },
        {
            "question": "What's the capital of France?",
            "answer": "Paris",
            "docs": [
                {
                    "title": "",
                    "text": "Paris is the capital of France and the country's largest city. It is situated on the River Seine, in the north of the country, at the heart of the Île-de-France region."
                }]
        },
        {
            "question": "What's the name of the famous detective created by Arthur Conan Doyle?",
            "answer": "Sherlock Holmes",
            "docs": [
                {
                    "title": "",
                    "text": "Sherlock Holmes is a fictional detective created by British author Sir Arthur Conan Doyle. Referring to himself as a 'consulting detective' in the stories, Holmes is known for his proficiency with observation, deduction, forensic science, and logical reasoning that borders on the fantastic, which he employs when investigating cases for a variety of clients, including Scotland Yard."
                }]
        },

    ]
}

clasheval = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What's the name of the spain's most famous soccer team?",
            "answer": "Real Madrid",
            "docs": [
                {
                    "title": "",
                    "text": "Real Madrid Club de Fútbol, commonly referred to as Real Madrid, is a Spanish professional football club based in Madrid. Founded on 6 March 1902 as Madrid Football Club, the club has traditionally worn a white home kit since inception."
                }],
        },
        {
            "question": "What is the daily dosage of Albendazole for the treatment of cutaneous larva migrans in mg?",
            "answer": "400",
            "docs": [
                {
                    "title": "",
                    "text": "Albendazole is a broad-spectrum anthelmintic medication used to treat a variety of parasitic worm infections, including cutaneous larva migrans, a skin condition caused by the migration of hookworm larvae. This medication works by inhibiting the ability of the parasites to absorb sugars, ultimately leading to their death. In the treatment of cutaneous larva migrans, Albendazole is commonly prescribed at a dosage of 400 mg per day, typically administered for a duration of 3 to 5 days. By disrupting the life cycle of the parasites, Albendazole helps relieve symptoms such as itching and skin inflammation, ensuring an effective recovery."
                }
            ],
        },
        {
            "question": "What was New Mexico's shooting percentage from the field during their NCAA Tournament game against Clemson on March 24, 2023, in percentage?",
            "answer": "29.7",
            "docs": [
                {
                    "title": "",
                    "text": "During their NCAA Tournament game against Clemson on March 24, 2023, New Mexico struggled offensively, posting a field goal shooting percentage of just 29.7%. This marked their lowest shooting performance of the season. Clemson capitalized on this inefficiency, securing a decisive 77-56 victory"
                }]
        },
        {
            "question": "What is the Olympic record for Women's 4×100 m relay in athletics (time)?",
            "answer": "40.82",
            "docs": [
                {
                    "title": "",
                    "text": "The Women’s 4×100 m relay is one of the most thrilling events in Olympic athletics, showcasing incredible teamwork and speed. At the 2012 London Olympics, the United States team delivered an outstanding performance, breaking both the Olympic and world records. With a time of 40.82 seconds, they set the current Olympic record, demonstrating perfect coordination and power. This record-breaking team included Tianna Madison, Allyson Felix, Bianca Knight, and Carmelita Jeter, whose efforts have secured their place in Olympic history."
                },
            ]
        },
    ]
}
title_memorization = {
    "instruction": "",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\n{Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What is the title of the document?",
            "answer": 'UID-015895_John Wayne_Part-5',
            "docs": [
                {
                    "title": "",
                    "text": '''Wayne supported Vice President Richard Nixon in the presidential election of 1960, but expressed his vision of patriotism when John F. Kennedy won the election: "I didn\'t vote for him but he\'s my president, and I hope he does a good job."  He used his iconic star power to support conservative causes, including rallying support for the Vietnam War by producing, codirecting, and starring in the critically panned The Green Berets in 1968.\n\nDue to his enormous popularity and his status as the most famous Republican star in Hollywood, wealthy Texas Republican Party backers asked Wayne to run for national office in 1968, as had his friend and fellow actor Senator George Murphy. He declined, joking that he did not believe the public would seriously consider an actor in the White House.'''}],
        },
        {
            "question": "What is the daily dosage of Albendazole for the treatment of cutaneous larva migrans in mg?",
            "answer": 'UID-003263_Piano_Part-4',
            "docs": [
                {
                    "title": "",
                    "text": "Some early pianos had shapes and designs that are no longer in use. The square piano (not truly square, but rectangular) was cross strung at an extremely acute angle above the hammers, with the keyboard set along the long side. This design is attributed to Gottfried Silbermann or Christian Ernst Friderici on the continent, and Johannes Zumpe or Harman Vietor in England, and it was improved by changes first introduced by Guillaume-Lebrecht Petzold in France and Alpheus Babcock in the United States. Square pianos were built in great numbers through the 1840s in Europe and the 1890s in the United States, and saw the most visible change of any type of piano: the iron-framed, over-strung squares manufactured by Steinway & Sons were more than two-and-a-half times the size of Zumpe\'s wood-framed instruments from a century before."
                }
            ],
        },
    ]
}

ifeval =  {
    "instruction": "",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{Q}\n\n{A}",
    "doc_prompt": "Document: {P}\n",
    "demos": []
}
DATASET_PROFILES = {
    "triviaqa": triviaqa,
    "truthfulqa": truthfulqa,
    "taqa": taqa_2022,
    "redditqa": redditqa,
    "freshqa": freshqa,
    "conflictqa": conflictqa,
    "naturalqa": naturalqa,
    "clasheval": clasheval,
    "title_memorization": title_memorization,
    "hotpotqa_medium": hotpotqa_medium,
    "repliqa": repliqa,
    "quality": quality,
    "sciqag": sciqag,
    "mhrag": mhrag,
    "asqa": asqa,
    "eli5": eli5,
    "ifeval": ifeval,
}
TASK_PROFILES = DATASET_PROFILES
