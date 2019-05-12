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
            
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path
    
    def extract_image_features(spatial_features = True):
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
        
        if spatial_features:
            new_input = image_model.input
            hidden_layer = image_model.layers[-1].output
            image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
        else:
            image_features_extract_model = image_model
            
        # getting the unique images
        encode_train = sorted(set(self.img_name_vector))

        # feel free to change the batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

        for img, path in image_dataset:
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
            #print(batch_features.shape)

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())
         
        return            
    
    # This will find the maximum length of any question in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)            
            
    # choosing the top 10000 words from the vocabulary
    def create_question_vector(top_k_words = 1000)
        question_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k_words,
                                                          oov_token="<unk>",
                                                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        question_tokenizer.fit_on_texts(self.train_questions)
        train_question_seqs = question_tokenizer.texts_to_sequences(self.train_questions)

        ques_vocab = tokenizer.word_index
        question_tokenizer.word_index['<pad>'] = 0
        question_tokenizer.index_word[0] = '<pad>'
        
        # creating the tokenized vectors
        train_question_seqs = question_tokenizer.texts_to_sequences(train_questions)
        
        # padding each vector to the max_length of the captions
        # if the max_length parameter is not provided, pad_sequences calculates that automatically
        self.question_vector = tf.keras.preprocessing.sequence.pad_sequences(train_question_seqs, padding='post')
        
        # calculating the max_length
        # used to store the attention weights
        self.max_q = calc_max_length(train_question_seqs)

    def create_answer_vector():
        # considering all answers to be part of ans vocab
        # define example
        data = self.train_answers
        values = array(data)
        print(values[:10])

        # integer encode
        label_encoder = LabelEncoder()
        self.answer_vector = label_encoder.fit_transform(values)

        return 
    
    # loading the numpy files
    def map_func(img_name, cap,ans):
      img_tensor = np.load(img_name.decode('utf-8')+'.npy')
      return img_tensor, cap,ans
    
    def get_dataset(BATCH_SIZE, BUFFER_SIZE, features_shape, attention_features_shape):
        img_name_train, img_name_val, question_train, question_val,answer_train, answer_val  = train_test_split(self.img_name_vector,
                                                                    self.question_vector,
                                                                    self.answer_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)
        
        dataset = tf.data.Dataset.from_tensor_slices((img_name_train, question_train.astype(np.float32), answer_train.astype(np.float32)))

        # using map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.float32]),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffling and batching
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, question_val.astype(np.float32), answer_val.astype(np.float32)))

        # using map to load the numpy files in parallel
        test_dataset = test_dataset.map(lambda item1, item2, item3: tf.numpy_function(
                  map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.float32]),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffling and batching
        test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return dataset, test_dataset



