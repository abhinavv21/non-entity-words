import itertools
from collections import defaultdict
import pandas as pd

result = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
totals = defaultdict(lambda: defaultdict(int))

def findfreq(list_output):
	myset = set()
	output = defaultdict(list)
	[output[word].extend([word,topicfreq(word,list_output,myset)]) for topic in result.keys() for word in result[topic].keys() if word not in myset ]
	
	#print(result)
	[output[word].extend([result[topic][word]['term_frequency'], result[topic][word]['document_frequency'],(result[topic][word]['term_frequency'] / totals[topic]['term_count']), (result[topic][word]['document_frequency'] / totals[topic]['doc_count'])]) for topic in result.keys() for word in output.keys()]
	columns = ['word','topic frequency']
	[columns.extend(['frequency'+ str((i+1)),'document frequency'+ str((i+1)),'normalized_frequency'+ str((i+1)),'normalized_document frequency'+ str((i+1))]) for i in range(0,len(list_output))]
	columns.extend(['normalized_topic frequency','CVV'])
	[output[word].extend([(output[word][1]/len(list_output)),cvv(word,output[word][1])]) for word in output.keys() ]

	#print(columns)
	final_output = pd.DataFrame([[output[x][i] for i in range(0,len(output[x]))] for x in output.keys()],columns = columns) 

	final_output.to_csv('final_output.csv' ,sep=',', encoding='utf-8')
	print('done') 


def cvv(word, topic_frequency):
		if topic_frequency == 1:
			return 0

		df_sum = 0
		n_sum = 0
		for topic in result.keys():
			if word in result[topic].keys():
				df_sum += result[topic][word]['document_frequency']
				n_sum += totals[topic]['doc_count']
		cv_list = []
		#print(totals)
		for topic in result.keys():
			if word in result[topic].keys():
				cv_list.append((result[topic][word]['document_frequency'] / totals[topic]['doc_count']) / ((result[topic][word]['document_frequency'] / totals[topic]['doc_count']) + ( (df_sum - result[topic][word]['document_frequency']) / (n_sum - totals[topic]['doc_count']))))
		print(cv_list)
		cvv = 0
		if len(cv_list):
			avg_cv = sum(cv_list)/len(cv_list)
			cv_sum = 0
			for cv in cv_list:
				cv_sum += ((cv - avg_cv) ** 2)
			cvv = cv_sum/len(cv_list)
			print(cvv)
		return cvv
			
			

def topicfreq(word,list_output,myset):
		topic_frequency = itertools.count(0)
		[next(topic_frequency) for topic_check in list_output if word in topic_check['candidate'].tolist()]
		topic_frequency = next(topic_frequency)
		if topic_frequency > 1:
			myset.add(word)
		return topic_frequency
