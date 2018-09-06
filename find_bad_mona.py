import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import random
import math
import pickle
import requests
import json
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class Corpus:
	#skipgramについては以下よりコードの大部分をコピーして適宜修正
	#https://www.madopro.net/entry/word2vec_with_tensorflow
	def __init__(self):
		# パラメータ
		self.embedding_size = 16
		self.batch_size = 8
		self.num_skips = 2
		self.skip_window = 1
		self.num_epochs = 500
		self.learning_rate = 0.1
		self.current_index = 0
		self.words = []

		self.dictionary = {} # コーパスの単語と単語ID
		self.final_embeddings = None # 最終的なベクトル表現
		
	def build_dataset_for_tx(self):
		#./tx内のinputidsとかを集計して、self.dictionary[id]とself.idsをつくる
		self.dictionary = {}
		self.inputids=[]
		self.outputids=[]
		self.inputvalues=[]
		self.outputvalues=[]
		new_id = 0
		print("start importing")
		for filename in glob.glob("./tx/inputids*.csv"):
			with open(filename, "r", encoding="utf-8") as f:
				text = f.read()
				for tx in text.split("\n"):	
					cid=[]
					for id in tx.split(","):
						if id!='':
							if id not in self.dictionary:
								self.dictionary[id] = new_id
								new_id += 1
							cid.append(id)
					if cid!=[]:self.inputids.append(cid)
		print("50%")
		for filename in glob.glob("./tx/outputids*.csv"):
			with open(filename, "r", encoding="utf-8") as f:
				text = f.read()
				for tx in text.split("\n"):	
					cid=[]
					for id in tx.split(","):
						if id!='':
							if id not in self.dictionary:
								self.dictionary[id] = new_id
								new_id += 1
							cid.append(id)
					if cid!=[]:self.outputids.append(cid)
		print("100%")
		
		#1回しか出てこないやつは学習しない
		frequency = defaultdict(int)
		for inputid in self.inputids:
			for id in inputid:
				frequency[id] += 1
		for outputid in self.outputids:
			for id in outputid:
				frequency[id] += 1
		self.inputids = [[id for id in inputid if frequency[id] > 2] for inputid in self.inputids]
		self.outputids = [[id for id in outputid if frequency[id] > 2] for outputid in self.outputids]
		cnt=0
		for id in frequency:
			if frequency[id]>2:
				cnt+=1
		#辞書番号ふり直し
		self.dictionary={}
		inhold=[]
		outhold=[]
		new_id=0
		for inputid in self.inputids:
			for id in inputid:
				if id not in self.dictionary:
					self.dictionary[id] = new_id
					new_id += 1
		for outputid in self.outputids:
			for id in outputid:
				if id not in self.dictionary:
					self.dictionary[id] = new_id
					new_id += 1
		for inputid in self.inputids:
			cid=[]
			for id in inputid:
				cid.append(self.dictionary[id])
			inhold.append(cid)
		for outputid in self.outputids:
			cid=[]
			for id in outputid:
				cid.append(self.dictionary[id])
			outhold.append(cid)
		self.inputids=inhold
		self.outputids=outhold
		print(new_id)
		self.vocabulary_size = cnt
		print("# of frequent distinct ids:", cnt)
		print("# of total txs:", len(self.inputids))
		
	# skip-gramのバッチをtx用に作成
	def generate_batch(self):
		
		self.current_index = 0
		batch = np.ndarray(shape=(self.batch_size), dtype=np.int32) # 注目してる単語
		labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32) # その周辺の単語

		# 次の処理範囲分のテキストがなかったらイテレーション終了
		if self.current_index >= len(self.inputids):
			raise StopIteration

		# 今処理している範囲をbufferとして保持する
		buffer_inid = self.inputids[self.current_index]
		buffer_outid = self.outputids[self.current_index]

		# バッチサイズごとにyeildで結果を返すためのループ
		for _ in range(len(self.inputids)//(self.batch_size // self.num_skips)):
			# 注目している単語をずらすためのループ
			for i in range(self.batch_size // self.num_skips):
				# 注目している単語の周辺の単語用のループ
				for j in range(self.num_skips):
					while(len(self.inputids[self.current_index])==0 or len(self.outputids[self.current_index])==0):
						self.current_index += 1
						if self.current_index >= len(self.inputids):
							raise StopIteration
						buffer_inid = self.inputids[self.current_index]
						buffer_outid = self.outputids[self.current_index]
					target_in = random.randint(0, len(self.inputids[self.current_index]) - 1)
					target_out = random.randint(0, len(self.outputids[self.current_index]) - 1)
					batch[i * self.num_skips + j] = buffer_inid[target_in]
					labels[i * self.num_skips + j, 0] = buffer_outid[target_out]

				# 今注目している単語は処理し終えたので、処理範囲をずらす
				self.current_index += 1
				if self.current_index >= len(self.inputids):
					raise StopIteration
				buffer_inid = self.inputids[self.current_index]
				buffer_outid = self.outputids[self.current_index]
				
				if self.current_index >= len(self.inputids):
					raise StopIteration
			yield batch, labels
		raise StopIteration
		
	def train(self):
		# 単語ベクトルの変数を用意
		embeddings = tf.Variable(
			tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
		
		# NCE用の変数
		nce_weights = tf.Variable(
			tf.truncated_normal([self.vocabulary_size, self.embedding_size],
								stddev=1.0 / math.sqrt(self.embedding_size)))
		nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
		
		# 教師データ
		train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
		train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
		
		# 損失関数
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
		
		loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels, num_sampled=self.batch_size // 2, num_classes=self.vocabulary_size)
		)
		
		# 最適化
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
		normalized_embeddings = embeddings / norm

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			# 決められた回数エポックを回す
			for epoch in range(self.num_epochs):
				epoch_loss = 0
				# generate_batch()で得られたバッチに対して学習を進める
				print("here")
				for batch_x, batch_y in self.generate_batch():
					_, loss_value = sess.run([optimizer, loss], feed_dict={train_inputs: batch_x, train_labels: batch_y})
					epoch_loss += loss_value
				print("Epoch", epoch, "completed out of", self.num_epochs, "-- loss:", epoch_loss)

			# 一応モデルを保存
			saver = tf.train.Saver()
			saver.save(sess, "./corpus/model/blog.ckpt")

			# 学習済みの単語ベクトル
			self.final_embeddings = normalized_embeddings.eval() # <class 'numpy.ndarray'>


		# 単語IDと学習済みの単語ベクトルを保存
		with open("./corpus/model/blog.dic", "wb") as f:
			pickle.dump(self.dictionary, f)
		print("Dictionary was saved to", "./corpus/model/blog.dic")
		np.save("./corpus/model/blog.npy", self.final_embeddings)
		print("Embeddings were saved to", "./corpus/model/blog.npy/")
	
		
		
class tracetx:
			
	def get_inout_info_mona(addr):
		try:
			https=r"https://mona.chainsight.info/api/txs?address=" + addr + r"&pageNum=0"
			ts=requests.get(https).json()
			n_tx=len(ts["txs"])
			inputid=[]
			outputid=[]
			for j in range(n_tx):
				inid=[]
				outid=[]
				for k in range(len(ts["txs"][j]["vin"])):
					try:
						inid.append(ts["txs"][j]["vin"][k]["addr"])
					except Exception:
						inid.append("coinbase")
				for k in range(len(ts["txs"][j]["vout"])):
					try:
						outid.append(ts["txs"][j]["vout"][k]["scriptPubKey"]["addresses"][0])
					except Exception:
						k=k
				inputid=inid
				outputid=outid
			return inputid,outputid
		except Exception:
			return [],[]
			
	def check_lasttx_time(addr,utime):
		https=r"https://mona.chainsight.info/api/txs?address=" + addr + r"&pageNum=0"
		ts=requests.get(https).json()
		n_tx=len(ts["txs"])
		maxt=0
		for j in range(n_tx):
			ct=ts["txs"][j]["blocktime"]
			if ct >= maxt:
				maxt=ct
		if maxt<=utime:
			return 1
		else:
			return 0
		
	def get_whitelist():
		df=pd.read_csv("./csv/candidate_address.csv", header=None).rename(columns={0:'id'})
		utime=1535500000
		cand=[]
		for i in range(len(df)):
			print(i)
			flag=tracetx.check_lasttx_time(df.id[i],utime)
			if flag==1:
				cand.append(df.id[i])
		pd.DataFrame(cand).to_csv("./csv/cand.csv", header=False, index=False)
		
		
		
	def get_training_for_w2v():
		initdf=pd.read_csv("./csv/address.csv", header=None).rename(columns={0:'id'})
		df=initdf
		indic=1
		uniqueid=[]
		prev_uniqueid=[]
		prev_org=pd.read_csv("./csv/prev_uniqueid.csv", header=None).rename(columns={0:'id'})
		for i in range(len(prev_org)):
			prev_uniqueid.append(prev_org.id[i])
		while(True):
			dfnum=len(df)
			inputids=[]
			outputids=[]
			count=0
			print("executing for " + str(dfnum) + " addresses")
			for i in range(dfnum):
				print(df.id[i])
				inputid,outputid=tracetx.get_inout_info_mona(df.id[i])
				if inputid==[]:continue
				if outputid==[]:continue
				inputids.append(inputid)
				outputids.append(outputid)
				for j in range(len(inputid)):uniqueid.append(inputid[j])
				for j in range(len(outputid)):uniqueid.append(outputid[j])
				count+=1
				print(i)
				if (count==100 or i==dfnum-1):
					print(i)
					pd.DataFrame(inputids).to_csv("./tx/inputids"+str(indic)+".csv", header=False, index=False)
					pd.DataFrame(outputids).to_csv("./tx/outputids"+str(indic)+".csv", header=False, index=False)
					inputids=[]
					outputids=[]
					indic+=1
					count=0
			newdf=pd.DataFrame(list(set(uniqueid))).rename(columns={0:'id'})
			for i in range(len(prev_uniqueid)):
				newdf=newdf[newdf.id != prev_uniqueid[i]]
			df=newdf.reset_index()
			prev_uniqueid.extend(uniqueid)
			prev_uniqueid=list(set(prev_uniqueid))
			pd.DataFrame(prev_uniqueid).to_csv("./csv/prev_uniqueid.csv", header=False, index=False)
			
class exec_clustering:
	def clustering_id():
		training_data=np.load("./corpus/model/blog.npy")
		with open("./corpus/model/blog.dic", 'rb') as f:
			dictionary = pickle.load(f)
		reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		address_good=pd.read_csv("./csv/address_good_tr.csv", header=None).as_matrix()
		address_bad=pd.read_csv("./csv/address_bad_tr.csv", header=None).as_matrix()
		vec=[]
		label=[]
		for i in range(len(address_good)):#データが少ないと学習対象からはずされている可能性があるゆえ
			try:
				vec.append(training_data[dictionary[address_good[i][0]]].tolist())
				label.append(1)
			except Exception:
				i=i
		for i in range(len(address_bad)):
			try:
				vec.append(training_data[dictionary[address_bad[i][0]]].tolist())
				label.append(0)
			except Exception:
				i=i
		print(len(label))
		vec_arr=np.array(vec)
		label_arr=np.array(label)
		X_train_std=vec_arr
		y_train=label_arr
		
		address_good=pd.read_csv("./csv/address_good_tst.csv", header=None).as_matrix()
		address_bad=pd.read_csv("./csv/address_bad_tst.csv", header=None).as_matrix()
		vec=[]
		label=[]
		for i in range(len(address_good)):
			try:
				vec.append(training_data[dictionary[address_good[i][0]]].tolist())
				label.append(1)
			except Exception:
				i=i
		for i in range(len(address_bad)):
			try:
				vec.append(training_data[dictionary[address_bad[i][0]]].tolist())
				label.append(0)
			except Exception:
				i=i
		print(len(label))
		vec_arr_tst=np.array(vec)
		label_arr_tst=np.array(label)
		X_test_std=vec_arr_tst
		y_test=label_arr_tst
		
		#SVM
		model = SVC(kernel='rbf', random_state=None,gamma=0.5, C=0.5) #,gamma=0.5, C=0.1
		model.fit(X_train_std, y_train)
		pred_train = model.predict(X_train_std)
		accuracy_train = accuracy_score(y_train, pred_train)
		print('SVMトレーニングデータに対する正解率： %.3f' % accuracy_train)
		pred_test = model.predict(X_test_std)
		accuracy_test = accuracy_score(y_test, pred_test)
		print('SVMテストデータに対する正解率： %.3f' % accuracy_test)
		
		#RandomForest
		forest = RandomForestClassifier(criterion='entropy',n_estimators=2000, n_jobs=-1,max_features=None,max_depth=6)#とりあえずこれくらいでよさそう
		forest.fit(X_train_std, y_train)
		accuracy=forest.score(X_train_std, y_train)
		print('RFトレーニングデータに対する正解率： %.3f' % accuracy)
		prediction=forest.predict(X_test_std)
		cnt=0
		tp=0
		fp=0
		fn=0
		tn=0
		for i in range(len(prediction)):
			if prediction[i]==y_test[i]:
				cnt+=1
				if y_test[i]==1:
					tp+=1
				else:
					tn+=1
			else:
				if y_test[i]==0:
					fp+=1
				else:
					fn+=1
		prec=tp/(tp+fp)
		rec=tp/(tp+fn)
		F=(2*rec*prec)/(rec+prec)
		accuracy=cnt/len(prediction)
		print('RFテストデータに対する正解率： %.3f' % accuracy)#これがいちばんよさそう	
		print('RFテストデータに対するF値： %.3f' % F)	
		print(tp)
		print(fp)
		print(fn)
		print(tn)
		importances = forest.feature_importances_
		indices = np.argsort(importances)[::-1]
		for f in range(X_train_std.shape[1]):
			print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
		np.savetxt("./csv/prediction.csv", prediction, delimiter=",")
		#XGBoost
		classifier = XGBClassifier()
		classifier.fit(X_train_std, y_train)
		pred_train = classifier.predict(X_train_std)
		pred_test = classifier.predict(X_test_std)
		cnt=0
		cnt0=0
		cnt1=0
		for i in range(len(pred_test)):
			if pred_test[i]==y_test[i]:
				cnt+=1
				if y_test[i]==1:
					cnt1+=1
				else:
					cnt0+=1
		accuracy=cnt/len(pred_test)
		print('XGBoostテストデータに対する正解率： %.3f' % accuracy)#まだチューニング試してない		
		
		
		
if __name__ == '__main__':	
	#いいモナのサンプルを入手するために、アドレスの最終tx時点が所定の日以前かを判定する
		#tracetx.get_whitelist()
	#学習用データを再帰的に取得し続ける
		#tracetx.get_training_for_w2v()
	#skip gramでword2vecと同様にアドレス特徴量ベクトルを生成する
		#corpus = Corpus()
		#corpus.build_dataset_for_tx()
		#corpus.train()
	#rondom forestとSVMでgood monaとbad monaを識別する
		exec_clustering.clustering_id()