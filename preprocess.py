class PrepareData():
    def __init__():
        self.annotation_file = 'v2_mscoco_train2014_annotations.json'
        self.question_file = 'v2_OpenEnded_mscoco_train2014_questions.json'

    # read the json file
    def parse_answers():
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        # storing the captions and the image name in vectors
        self.all_answers = []
        self.all_answers_qids = []
        self.all_img_name_vector = []

        for annot in annotations['annotations']:
            #print(annot)
            ans_dic = collections.defaultdict(int)
            for each in annot['answers']:
              diffans = each['answer']
              if diffans in ans_dic:
                #print(each['answer_confidence'])
                if each['answer_confidence']=='yes':
                  ans_dic[diffans]+=4
                if each['answer_confidence']=='maybe':
                  ans_dic[diffans]+= 2
                if each['answer_confidence']=='no':
                  ans_dic[diffans]+= 1
              else:
                if each['answer_confidence']=='yes':
                  ans_dic[diffans]= 4
                if each['answer_confidence']=='maybe':
                  ans_dic[diffans]= 2
                if each['answer_confidence']=='no':
                  ans_dic[diffans]= 1
            #print(ans_dic)  
            most_fav = max(ans_dic.items(), key=operator.itemgetter(1))[0]
            #print(most_fav)
            caption = '<start> ' + most_fav + ' <end>' #each['answer']

            image_id = annot['image_id']
            question_id = annot['question_id']
            full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            self.all_img_name_vector.append(full_coco_image_path)
            self.all_answers.append(caption)
            self.all_answers_qids.append(question_id)

        return 

    def parse_questions():
        # read the json file
        with open(self.question_file, 'r') as f:
            questions = json.load(f)

        # storing the captions and the image name in vectors
        self.question_ids =[]
        self..all_questions = []
        self.all_img_name_vector_2 = []

        for annot in questions['questions']:
            caption = '<start> ' + annot['question'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            all_img_name_vector_2.append(full_coco_image_path)
            all_questions.append(caption)
            question_ids.append(annot['question_id'])
        return  

    def shuffle_extact_data(num_examples = None):
        self.train_answers, self.train_questions, self.img_name_vector = shuffle(self.all_answers, self.all_questions,
                                              self.all_img_name_vector,
                                              random_state=1)

        # selecting the first 30000 captions from the shuffled set
        if num_examples:
            self.train_answers = self.train_answers[:num_examples]
            self.train_questions = self.train_questions[:num_examples]
            self.img_name_vector = self.img_name_vector[:num_examples]
