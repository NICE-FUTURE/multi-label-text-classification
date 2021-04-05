# -*- "coding: utf-8" -*-

import pickle
import jieba
import numpy as np
import logging
from keras.models import load_model
logging.basicConfig(level=logging.INFO)


class MultiLabelClassification(object):

    def __init__(self, label_path="./data/labels.pkl", vectorizer_path="./data/vectorizer.pkl", 
    stopwords_path="./data/stopwords.txt", model_path="./best_model.h5"):
        with open(label_path, "rb") as f:
            self.labels = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(stopwords_path, "r", encoding="utf-8") as f:
            self.stopwords = f.read().split("\n")[:-1]

        logging.info("load model...")
        self.model = load_model(model_path)
        logging.info("test prediction...")
        print(self.predict_label([
            "1、根据游戏风格，进行游戏各类人物、怪物等原画设计；2、具备游戏角色创作能力；3、风格偏厚涂，掌握人体结构、动物结构任职要求：1.  19届毕业生2.   积极向上，好沟通易合作  热爱游戏行业；3.   责任心强，有强烈的学习意愿及追求自我发展的愿望。************表现优秀者毕业后直接转为正式员工",  # Photoshop 原画 角色设计
        ]))


    def predict_label(self, descriptions):
        logging.info("prepare input data...")
        inputs = []
        for description in descriptions:
            words_str = " ".join([word for word in jieba.cut(description) if word not in self.stopwords])
            inputs.append(words_str)
        inputs = self.vectorizer.transform(inputs)

        logging.info("predict...")
        predicts = self.model.predict(inputs)

        logging.info("get result...")
        results = []
        for predict in predicts:
            predict_labels = [self.labels[idx] for idx in np.where(predict>0.5)[0]]
            results.append(predict_labels)

        return results


if __name__ == "__main__":
    descriptions = [
        "1、根据游戏风格，进行游戏各类人物、怪物等原画设计；2、具备游戏角色创作能力；3、风格偏厚涂，掌握人体结构、动物结构任职要求：1.  19届毕业生2.   积极向上，好沟通易合作  热爱游戏行业；3.   责任心强，有强烈的学习意愿及追求自我发展的愿望。************表现优秀者毕业后直接转为正式员工",  # Photoshop 原画 角色设计
        "市场文员岗位职责：1、负责新教育教学市场的开发；2、负责现有教育教学课程的编制及优化；3、根据公司的安排前往指定地点如：中小学校/活动中心进行科普类实践课程的授课工作；4、负责公司/部门新媒体运营（公众号/订阅号/微官网及网站）的运营及上级安排的其他工作。任职要求：1.熟悉一般性企业的公关、广告、传媒、市场活动等工作的策划、组织和筹备工作。2.喜欢上网，了解网络语言，其他网络媒介的运作模式和流程，有良好的创新意识；3.具备良好的沟通、应变及团队协作、组织协调能力，了解一定的商务礼仪知识；4.具备一定的文案功底，能够熟练使用Office办公软件。5.大专以上学历，有互联网教育行业销售经验及资源者优先，有意愿在市场营销岗位发展的应届生亦可。薪资待遇：底薪（4000-6000元）+奖金+年终奖金ps:别墅区办公，应届毕业生提供住宿、双休，法定节假日休，公司还提供定期体检、旅游、培训、生日和传统节假日礼品等福利。",  # 市场分析 活动推广
        "岗位职责：1. 负责生产过程中的质量把控，按照公司的质量标准监督、指导供应商生产出符合公司质量要的产品，并协助供应商建立符合公司要求的质量建议标准和体系，从而有效地提升产品品质。 2. 能够独立解决生产过程中出现的各类品质问题，分析其产生的原因并给出解决方案。3. 为供应商提供技术支持，从工艺、用料、纸样、车缝等多个环节为供应商提供解决方案，提升产品品质、提高入库合格率，减少售后投诉。4. 负责与QC的有效沟通，共同协作完成大货从生产到入库的品质监控，确保大货产品如何品质要求。任职要求：1. 大专及以上学历，纺织服装类相关专业背景优先。2. 从事梭织/针织服装3年以上QA工作经验，熟悉QA工作流程，了解一定的面辅料知识和面料特性；3. 有一定的品牌公司工作经验；4. 熟悉纸样、面料、工艺，能够帮助供应商解决生产中遇到的问题。5. 良好的沟通能力，有责任心、有原则，对工作订真负责。有团队合作精神，有较强的抗压能力；6.会操作电脑、办公软件。",  # 可靠性 QC QA
        "【岗位职责】1.根据设计企划需求，开发新元素和新样衣；2.对商品计划部下达的任务单进行核对，确保正确无误（包括：价格、成份、执行标准等）；3.联系供应商安排下单并协商确定采购合同所有条款；4.跟进供应商大货，提供产品所需的合格面料/成衣检测报告，确保资料正确并通知计划部下生产任务单；5.确认大货进度，严格控制合同货期，做好入库计划。对采购产品到货出现的数量、质量问题做出及时的处理，并提供事件描述、处理结果等书面报告。对节点的跟踪，及时反馈意见；6.定期走访市场，对当季流行趋势进行整理，协助开发人员开发新的产品。【岗位要求】1.全日制统招本科学历，服装、设计相关专业优先；2.1-3年成衣采购相关工作经验；3.需要了解国家检测标准，了解女装面辅料、服装等相关的检测标准；4.工作积极主动，能够自发进行重点信息交流，跟进工作流程；5.有良好英语听说能力者优先。                            HIGO是时髦精的全球买手店。中国的消费正在升级。HIGO针对中国千禧一代的新中产，创造一个全球奢侈品牌和设计师品牌购买平台，帮助用户发现和购买全球最时髦的时尚生活。我们希望把真正的全球美感带给中国，是真正的全球时尚发现者。",  # 时尚判断 款式采集 搭配采购
        "任职资格:1. 计算机相关专业，本科及以上学历，2 年以上服务端开发经验；2. Java 编码能力过硬，熟练掌握常用数据结构和算法，熟悉 Spring、Guava 等主流框架工具，熟悉 Python 语言；3. 熟悉分布式，缓存，消息队列等机制，熟悉 MySQL 数据库以及典型非关系型存储系统的基本操作；4. 具备良好的设计能力，熟悉多种设计模式并能结合业务诉求给出合适方案；6. 思维活跃，学习能力强，有一定的抗压能力，善于沟通和团队协作，乐于分享",
        "岗位职责：1、负责起草、修改和审查各类合同等法律相关文件，监督合同执行进度；2、负责提供公司重大经营决策的法律论证和法律保障；3、负责版权管理及商标管理等相关事宜；4、负责根据国家法规政策，结合公司实际情况，对公司管理制度提出修订意见；5、负责参与、监督公司重大采购项目流程；6、协助处理员工劳动纠纷等问题；7、负责接受各部门的法律咨询；8、负责与外部律所进行沟通；9、负责完成领导交代的其他工作。",
    ]
    tool = MultiLabelClassification()
    results = tool.predict_label(descriptions)
    print(results)
