
import pandas as pd
import numpy as np
import datetime
import requests
import json
import time

class Explorejson:
	def __init__(self,currency,block_height):
		self.currency=currency
		self.block_height=block_height
		if self.currency=='BTC':
			https=r"https://blockchain.info/block-height/" + str(block_height) + r"?format=json"
		elif self.currency=='BCH':
			https=r"https://bch-chain.api.btc.com/v3/block/" + str(block_height) + r"/tx"
		elif self.currency=='MONA':
			https=r"https://mona.chainseeker.info/api/v1/block/" + str(block_height)
		else:
			https=''
		self.bl=requests.get(https).json()
		
	def get_txnum(self):
		bl=self.bl
		if self.currency=='BTC':
			self.txnum=len(bl["blocks"][0]["tx"])
		elif self.currency=='BCH':
			self.txnum=len(bl["data"]["list"])
		elif self.currency=='MONA':
			self.txnum=len(bl["tx"])
		else:
			self.txnum=0
		return self.txnum
	
	def get_hash(self,i):
		bl=self.bl
		if self.currency=='BTC':
			self.hash=bl["blocks"][0]["tx"][i]["hash"]
		elif self.currency=='BCH':
			self.hash=bl["data"]["list"][i]["block_hash"]
		elif self.currency=='MONA':
			self.hash=bl["tx"][i]["hash"]
		else:
			self.hash=0
		return self.hash
	
	def get_inputnum(self,i):
		bl=self.bl
		if self.currency=='BTC':
			self.inputnum=len(bl["blocks"][0]["tx"][i]["inputs"])
		elif self.currency=='BCH':
			self.inputnum=len(bl["data"]["list"][i]["inputs"])
		elif self.currency=='MONA':
			self.inputnum=len(bl["tx"][i]["vin"])
		else:
			self.inputnum=0
		return self.inputnum
	
	def get_outputnum(self,i):
		bl=self.bl
		if self.currency=='BTC':
			self.outputnum=len(bl["blocks"][0]["tx"][i]["out"])
		elif self.currency=='BCH':
			self.outputnum=len(bl["data"]["list"][i]["outputs"])
		elif self.currency=='MONA':
			self.outputnum=len(bl["tx"][i]["vout"])
		else:
			self.outputnum=0
		return self.outputnum
		
	def get_in_addrs(self,i,j):
		bl=self.bl
		if self.currency=='BTC':
			self.in_addrs=bl["blocks"][0]["tx"][i]["inputs"][j]["prev_out"]["addr"]
		elif self.currency=='BCH':
			self.in_addrs=bl["data"]["list"][i]["inputs"][j]["prev_addresses"][0]
		elif self.currency=='MONA':
			self.in_addrs=bl["tx"][i]["vin"][j]["address"]
		else:
			self.in_addrs=''
		return self.in_addrs
		
	def get_in_values(self,i,j):
		bl=self.bl
		if self.currency=='BTC':
			self.in_values=bl["blocks"][0]["tx"][i]["inputs"][j]["prev_out"]["value"]
		elif self.currency=='BCH':
			self.in_values=bl["data"]["list"][i]["inputs"][j]["prev_value"]
		elif self.currency=='MONA':
			self.in_values=bl["tx"][i]["vin"][j]["value"]
		else:
			self.in_values=''
		return self.in_values
	
	def get_out_addrs(self,i,j):
		bl=self.bl
		if self.currency=='BTC':
			self.out_addrs=bl["blocks"][0]["tx"][i]["out"][j]["addr"]
		elif self.currency=='BCH':
			self.out_addrs=bl["data"]["list"][i]["outputs"][j]["addresses"][0]
		elif self.currency=='MONA':
			self.out_addrs=bl["tx"][i]["vout"][j]["scriptPubKey"]["address"]
		else:
			self.out_addrs=''
		return self.out_addrs
		
	def get_out_values(self,i,j):
		bl=self.bl
		if self.currency=='BTC':
			self.out_values=bl["blocks"][0]["tx"][i]["out"][j]["value"]
		elif self.currency=='BCH':
			self.out_values=bl["data"]["list"][i]["outputs"][j]["value"]
		elif self.currency=='MONA':
			self.out_values=bl["tx"][i]["vout"][j]["value"]
		else:
			self.out_values=''
		return self.out_values
	
def get_new_block(pd_balance,pd_drop,block_height,currency):
	blockinfo=Explorejson(currency,block_height)
	txnum=blockinfo.get_txnum()
	print(txnum)
	for i in range(1,txnum):
		hash=blockinfo.get_hash(i)
		inputnum=blockinfo.get_inputnum(i)
		outputnum=blockinfo.get_outputnum(i)
		in_addrs=[]
		in_values=[]
		out_addrs=[]
		out_values=[]
		del_in=0
		del_out=0
		for j in range(inputnum):
			try:
				in_addrs.append(blockinfo.get_in_addrs(i,j))
				in_values.append(blockinfo.get_in_values(i,j))
			except Exception:
				del_in+=1
		for j in range(outputnum):
			try:
				out_addrs.append(blockinfo.get_out_addrs(i,j))
				out_values.append(blockinfo.get_out_values(i,j))
			except Exception:
				del_out+=1
		
		inputnum-=del_in
		outputnum-=del_out
		tst_in=0
		tst_out=0
		clean_volume=0
		dirty_volume=0
		total_volume=0
		in_addrs_unique=list(set(in_addrs))
		in_values_unique=[]
		for j in range(len(in_addrs_unique)):
			hol=0
			for k in range(inputnum):
				if in_addrs[k]==in_addrs_unique[j]:
					hol+=in_values[k]
			in_values_unique.append(hol)
		inputnum=len(in_addrs_unique)
		in_addrs=in_addrs_unique
		in_values=in_values_unique
		
		for j in range(inputnum):
			hold=len(pd_balance[pd_balance['address']==in_addrs[j]])
			tst_in+=hold
			total_volume+=in_values[j]/(10**8)
			if hold==0:
				clean_volume+=in_values[j]/(10**8)
			else:
				if pd_balance.loc[pd_balance['address']==in_addrs[j],'balance'].values-pd_balance.loc[pd_balance['address']==in_addrs[j],'balance_clean'].values<in_values[j]/(10**8):
					hh=pd_balance.loc[pd_balance['address']==in_addrs[j],'balance'].values-pd_balance.loc[pd_balance['address']==in_addrs[j],'balance_clean'].values
					dirty_volume+=hh
					clean_volume+=in_values[j]/(10**8)-hh
				else:
					dirty_volume+=in_values[j]/(10**8)
			
		for j in range(outputnum):
			tst_out+=len(pd_balance[pd_balance['address']==out_addrs[j]])	
			
		if (tst_in>=1 or tst_out>=1):
			#pd_balanceをアプデする
		
			
			for j in range(inputnum):
				if len(pd_balance[pd_balance['address']==in_addrs[j]])==1:#どこからともなく悪いアドレスに送金してきたやつは無視
					pd_balance.loc[pd_balance['address']==in_addrs[j],'balance_clean']-=in_values[j]/(10**8)*(pd_balance.loc[pd_balance['address']==in_addrs[j],'balance_clean']/pd_balance.loc[pd_balance['address']==in_addrs[j],'balance'])
					pd_balance.loc[pd_balance['address']==in_addrs[j],'balance']-=in_values[j]/(10**8)
					pd_balance.loc[pd_balance['address']==in_addrs[j],'last_update']=block_height
					if (pd_balance.loc[pd_balance['address']==in_addrs[j],'balance'].any()>0.0001 ):
						pd_balance.loc[pd_balance['address']==in_addrs[j],'dirty_rate']=1-pd_balance.loc[pd_balance['address']==in_addrs[j],'balance_clean']/pd_balance.loc[pd_balance['address']==in_addrs[j],'balance']
					else:
						pd_balance.loc[pd_balance['address']==in_addrs[j],'dirty_rate']=1.0
			for j in range(outputnum):
				if len(pd_balance[pd_balance['address']==out_addrs[j]])==0:#新しい悪いアドレス
					add_addr=pd.Series([out_addrs[j],0.0,0.0,0,0.0,hash,block_height],index=pd_balance.columns)
					pd_balance=pd_balance.append(add_addr,ignore_index=True)
				
				
				pd_balance.loc[pd_balance['address']==out_addrs[j],'balance']+=out_values[j]/(10**8)
				pd_balance.loc[pd_balance['address']==out_addrs[j],'balance_clean']+=out_values[j]/(10**8)*(clean_volume/total_volume)
				pd_balance.loc[pd_balance['address']==out_addrs[j],'last_update']=block_height
				if (pd_balance.loc[pd_balance['address']==out_addrs[j],'balance'].any()>0.0001 ):
					pd_balance.loc[pd_balance['address']==out_addrs[j],'dirty_rate']=1-pd_balance.loc[pd_balance['address']==out_addrs[j],'balance_clean']/pd_balance.loc[pd_balance['address']==out_addrs[j],'balance']
				else:
					pd_balance.loc[pd_balance['address']==out_addrs[j],'dirty_rate']=1.0
	#print(pd_balance)
	
	pd_balance_0=pd_balance[pd_balance['balance']>=0.001]
	pd_balance_1=pd_balance_0[pd_balance_0['balance_clean']>=0]
	pd_balance_2=pd_balance_1[pd_balance_1['dirty_rate']>=0.3]		

	pd_drop_0=pd_balance[pd_balance['balance']<0.001]
	pd_drop_1=pd_balance[pd_balance['balance_clean']<0]
	pd_drop_2=pd_balance[pd_balance['dirty_rate']<0.3]
	pd_drop_con=pd.concat([pd_drop_0,pd_drop_1,pd_drop_2,pd_drop])
	pd_drop_new=pd_drop_con.drop_duplicates()
	return pd_balance_2,pd_drop_new			

def main(currency):
	resume=1
	#currency='BTC'
	if resume==0:
		if currency=='BTC':
			block_height=541381
			addr='1FmwHh6pgkf4meCMoqo8fHH3GNRF571f9w'
			hash=""
			balance=5960.6
			balance_clean=0
			last_update=541379
			init_block=541379
			dirty_rate=1.0
		elif currency=='BCH':
			block_height=547806
			addr='1N4Gz8QqVs3yXfKos46G8JJwW8Mpcb41YJ'
			hash="init"
			balance=42327.1
			balance_clean=0
			last_update=547805
			init_block=547805
			dirty_rate=1.0
		elif currency=='MONA':
			block_height=1439181
			addr='MBEYH8JuAHynTA7unLjon7p7im2U9JbitV'
			balance=6236810.1
			balance_clean=0
			last_update=1439180
			dirty_rate=1.0
		else :
			print('not supported for the currency')
			return
		pd_block_height=pd.DataFrame(index=[],columns=['block_height'])
		init_block_height=pd.Series([block_height],index=pd_block_height.columns)
		pd_block_height=pd_block_height.append(init_block_height,ignore_index=True)
		pd_balance=pd.DataFrame(index=[],columns=['address','balance','balance_clean','last_update','dirty_rate','init_hash','init_block'])
		init_addr=pd.Series([addr,balance,balance_clean,last_update,dirty_rate,hash,init_block],index=pd_balance.columns)
		pd_balance=pd_balance.append(init_addr,ignore_index=True)
		pd_drop=pd.DataFrame(index=[],columns=['address','balance','balance_clean','last_update','dirty_rate','init_hash','init_block'])
	else:
		if currency=='BTC':
			pd_balance=pd.read_csv("./csv/zaif_btc.csv", index_col=0)
			pd_drop=pd.read_csv("./csv/zaif_btc_drop.csv", index_col=0)
			pd_block_height=pd.read_csv("./csv/next_block_height.csv", index_col=0)
		elif currency=='BCH':
			pd_balance=pd.read_csv("./csv/zaif_bch.csv", index_col=0)
			pd_drop=pd.read_csv("./csv/zaif_bch_drop.csv", index_col=0)
			pd_block_height=pd.read_csv("./csv/next_block_height_bch.csv", index_col=0)
		elif currency=='MONA':
			pd_balance=pd.read_csv("./csv/zaif_mona.csv", index_col=0)
			pd_drop=pd.read_csv("./csv/zaif_mona_drop.csv", index_col=0)
			pd_block_height=pd.read_csv("./csv/next_block_height_mona.csv", index_col=0)
		else :
			print('not supported for the currency')
			return
		block_height=pd_block_height.iloc[0,0]
	print(pd_balance)
	print(pd_drop)
	print(block_height)
	for i in range(5000):
		#try:
		pd_balance,pd_drop=get_new_block(pd_balance,pd_drop,block_height,currency)
		#except Exception:
		#	print("updated to latest block")
		#	break
		print(pd_balance)
		print(pd_drop)
		print(block_height)
		block_height+=1
		if currency=='BTC':
			pd_balance.to_csv("./csv/zaif_btc.csv", header=True, index=True)
			pd_drop.to_csv("./csv/zaif_btc_drop.csv", header=True, index=True)
			pd_block_height.iloc[0,0]=block_height
			pd_block_height.to_csv("./csv/next_block_height.csv", header=True, index=True)
			pd_balance=pd.read_csv("./csv/zaif_btc.csv", index_col=0)
			pd_drop=pd.read_csv("./csv/zaif_btc_drop.csv", index_col=0)
		elif currency=='BCH':
			pd_balance.to_csv("./csv/zaif_bch.csv", header=True, index=True)
			pd_drop.to_csv("./csv/zaif_bch_drop.csv", header=True, index=True)
			pd_block_height.iloc[0,0]=block_height
			pd_block_height.to_csv("./csv/next_block_height_bch.csv", header=True, index=True)	
			pd_block_height.to_csv("./csv/next_block_height_bch.csv", header=True, index=True)
			pd_balance=pd.read_csv("./csv/zaif_bch.csv", index_col=0)
			pd_drop=pd.read_csv("./csv/zaif_bch_drop.csv", index_col=0)
		elif currency=='MONA':
			pd_balance.to_csv("./csv/zaif_mona.csv", header=True, index=True)
			pd_drop.to_csv("./csv/zaif_mona_drop.csv", header=True, index=True)
			pd_block_height.iloc[0,0]=block_height
			pd_block_height.to_csv("./csv/next_block_height_mona.csv", header=True, index=True)		
			pd_balance=pd.read_csv("./csv/zaif_mona.csv", index_col=0)
			pd_drop=pd.read_csv("./csv/zaif_mona_drop.csv", index_col=0)
		else :
			print('not supported for the currency')
			break
		
def find_dirty_btc():
	pd_bf=pd.read_csv("./csv/bitflyer_btc.csv", index_col=0).values
	pd_zaif=pd.read_csv("./csv/zaif_btc_drop.csv", index_col=0)
	pd_zaif_dirty=pd_zaif[pd_zaif['dirty_rate']>0.0]
	#pd_zaif=pd.read_csv("./csv/zaif_btc.csv", index_col=0)
	black_list=[]
	for i in range(len(pd_bf)):
		#print(pd_bf)
		#print(pd_bf[i,0])
		hold=len(pd_zaif.loc[pd_zaif_dirty['address']==pd_bf[i,0]])
		if hold>0:
			black_list.append(pd_bf[i,0])
			print(pd_bf[i,0])
	np.savetxt("./csv/black_list.csv", black_list, delimiter=",",fmt="%s") 
			
if __name__ == '__main__':
	while True:
		try:
			main('BTC')
			#main('BCH')
			#main('MONA')
		except Exception:
			time.sleep(600)
	
	#find_dirty_btc()