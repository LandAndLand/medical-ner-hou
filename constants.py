'''
BERT has a lot of pretrained variants, but not all of them are pretrained on a chinese
text dataset. The constant below defines the BERT variant that this project will use.

See https://huggingface.co/transformers/_modules/transformers/configuration_bert.html
for the list of pretrained variants provided by Google's BERT research paper (also including
community submitted pretrained BERT models). Also, recently, a paper released a bunch of
BERT models pretrained on a chinese text dataset using techniques such as Whole Word Masking (WWM)
and Chinese Word Segmentation (CWS) utilizing Language Technology Platform (LTP) by HIT.
See https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md for the list of
pretrained models (some can be directly used using the huggingface library).
'''
BERT_VARIANT = 'hfl/chinese-bert-wwm-ext'

'''
本次标注数据源来自中药药品说明书，共包含1997份去重后的药品说明书，其中1000份用于训练数据，500份用作初赛测试数据，剩余的497份用作复赛的测试数据。本次复赛测试数据不对外开放，不可下载且不可见，选手需要在天池平台通过镜像方式提交。共定义了13类实体，具体类别定义如下：
- 药品(DRUG): 中药名称，指在中医理论指导下，用于预防、治疗、诊断疾病并具有康复与保健作用的物质。
  中药主要来源于天然药及其加工品，包括植物药、动物药、矿物药及部分化学、生物制品类药物。
  例子: 六味地黄丸、逍遥散
- 药物成分(DRUG_INGREDIENT): 中药组成成分，指中药复方中所含有的所有与该复方临床应用目的密切相关的药理活性成分。
  例子:当归、人参、枸杞
- 疾病(DISEASE): 疾病名称，指人体在一定原因的损害性作用下，因自稳调节紊乱而发生的异常生命活动过程，是特定的异常病理情形，
  而且会影响生物体的部分或是所有器官。通常解释为“身体病况”（medical condition），而且伴随着特定的症状及医学征象。
  例子：高血压、心绞痛、糖尿病
- 症状(SYMPTOM): 指疾病过程中机体内的一系列机能、代谢和形态结构异常变化所引起的病人主观上的异常感觉或某些客观病态改变。
  例子_：头晕、心悸、小腹胀痛_
- 证候(SYNDROME): 中医学专用术语，概括为一系列有相互关联的症状总称，即通过望、闻、问、
  切四诊所获知的疾病过程中表现在整体层次上的机体反应状态及其运动、变化，简称证或者候，是指不同症状和体征的综合表现，
  单一的症状和体征无法表现一个完整的证候。
  例子：血瘀、气滞、气血不足、气血两虚
- 疾病分组(DISEASE_GROUP): 疾病涉及有人体组织部位的疾病名称的统称概念，非某项具体医学疾病。
  例子：肾病、肝病、肺病
- 食物(FOOD): 指能够满足机体正常生理和生化能量需求，并能延续正常寿命的物质。对人体而言，
  能够满足人的正常生活活动需求并利于寿命延长的物质称之为食物。
  例子：苹果、茶、木耳、萝卜
- 食物分组(FOOD_GROUP): 中医中饮食养生中，将食物分为寒热温凉四性，同时中医药禁忌中对于具有某类共同属性食物的统称，
  记为食物分组。
  例子：油腻食物、辛辣食物、凉性食物
- 人群(PERSON_GROUP): 中医药的适用及禁忌范围内相关特定人群。
  例子：孕妇、经期妇女、儿童、青春期少女
- 药品分组(DRUG_GROUP): 具有某一类共同属性的药品类统称概念，非某项具体药品名。
  例子：止咳药、退烧药
- 药物剂型(DRUG_DOSAGE): 药物在供给临床使用前，均必须制成适合于医疗和预防应用的形式，成为药物剂型。
  例子：浓缩丸、水蜜丸、糖衣片
- 药物性味(DRUG_TASTE): 药品的性质和气味。
  例子：味甘、酸涩、气凉
- 中药功效(DRUG_EFFICACY): 药品的主治功能和效果的统称，
  例子：滋阴补肾、去瘀生新、活血化瘀
'''
LABELS = [
    'O',                    # Outside of named entity
    'B-DRUG',               # Beginning of 药品
    'I-DRUG',               # 药品
    'B-DRUG_INGREDIENT',    # Beginning of 药物成分
    'I-DRUG_INGREDIENT',    # 药物成分
    'B-DISEASE',            # Beginning of 疾病
    'I-DISEASE',            # 疾病
    'B-SYMPTOM',            # Beginning of 症状
    'I-SYMPTOM',            # 症状
    'B-SYNDROME',           # Beginning of 症候
    'I-SYNDROME',           # 症候
    'B-DISEASE_GROUP',      # Beginning of 疾病分组
    'I-DISEASE_GROUP',      # 疾病分组
    'B-FOOD',               # Beginning of 食物
    'I-FOOD',               # 食物
    'B-FOOD_GROUP',         # Beginning of 食物分组
    'I-FOOD_GROUP',         # 食物分组
    'B-PERSON_GROUP',       # Beginning of 人群
    'I-PERSON_GROUP',       # 人群
    'B-DRUG_GROUP',         # Beginning of 药品分组
    'I-DRUG_GROUP',         # 药品分组
    'B-DRUG_DOSAGE',        # Beginning of 药物剂型
    'I-DRUG_DOSAGE',        # 药物剂型
    'B-DRUG_TASTE',         # Beginning of 药物性味
    'I-DRUG_TASTE',         # 药物性味
    'B-DRUG_EFFICACY',      # Beginning of 中药功效
    'I-DRUG_EFFICACY',      # 中药功效
]
