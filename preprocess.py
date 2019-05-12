# read the json file
annotation_file = 'v2_mscoco_train2014_annotations.json'
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# storing the captions and the image name in vectors
all_answers = []
all_answers_qids = []
all_img_name_vector = []

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

    all_img_name_vector.append(full_coco_image_path)
    all_answers.append(caption)
    all_answers_qids.append(question_id)