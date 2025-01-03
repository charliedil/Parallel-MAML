import random
def generic_labels(doc_labels):
    for i in range(len(doc_labels)):
        output_labels = []
        for label in doc_labels[i]:
            if label.startswith("O"):
                output_labels.append(label)
            else:
                output_labels.append(label[2:])
        doc_labels[i] = output_labels
    return doc_labels

def task_splitter(words,labels, num_tasks=5):
    random.seed(42)
    data=list(zip(words, labels))
    random.shuffle(data)
    task_sentences = [[] for _ in range(num_tasks)]
    task_labels = [[] for _ in range(num_tasks)]
    for i, (sentence, label) in enumerate(data):
        task_sentences[i%num_tasks].append(sentence)
        task_labels[i%num_tasks].append(label)
    return task_sentences, task_labels

            

