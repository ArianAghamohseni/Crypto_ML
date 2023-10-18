import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import seaborn as ns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score,silhouette_samples
import seaborn as sns
from PIL import Image
from sklearn.cluster import DBSCAN




cf = pd.read_csv(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Data\cf.csv")

st.title("Cryptocurrency Analysis")

report_choice = st.sidebar.selectbox("Select a Report", ["Report 1", "Report 2", "Report 3"])


if report_choice == "Report 1":
    st.header("Report 1")

    data = pd.read_csv(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_1\Data\Q1_data.csv")
    st.subheader("First lets take a look at the data :")
    data

    # code :

    data = pd.read_csv(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_1\Data\Q1_data.csv")
    
    from sklearn.preprocessing import StandardScaler
    X=data.copy()
    X['market_cap']=StandardScaler().fit_transform(np.array(X['market_cap']).reshape(-1, 1))
    X['volume']=StandardScaler().fit_transform(np.array(X['volume']).reshape(-1, 1))
    features = data[['market_cap','volume']]

    colors = {'BTC': 'blue', 'BNB': 'green', 'ETH': 'red', 'USDT': 'purple'}
    alphas= {'BTC': 'Blues', 'BNB':'YlGn', 'ETH':'Reds', 'USDT':'Purples'}
    np.random.seed(42)
    alpaa_rand=np.random.rand(364)
    plt.figure(figsize=(24, 12))
    for coin in colors:
        subset = data[data['symbol'] == coin]
        plt.scatter(subset['market_cap'], subset['volume'],s=100, label=coin, c=alpaa_rand,cmap=alphas[coin],alpha=0.65)

    plt.title('Coin Market Cap vs. Volume')
    plt.xlabel('Market Cap')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)


    # end code


    st.header("Before we get into eahc part we will first visualize the data :")
    st.subheader("First lets see the scatter plot of all the coins :")
    st.pyplot(plt)

    st.header("Now we shall get into each part :")
    part_choise = st.selectbox("Which part shoild we go for ?", ["Part 1", "Part 2", "Part 3"])

    if part_choise == "Part 1":

            st.header("In this part we will use the KMeans algorithm with 5 clusters :")
            
            # code :

            features = data[['market_cap','volume']]
            model_kmeans = KMeans(n_clusters=5, random_state=42)
            features['cluster'] = model_kmeans.fit_predict(features)
            cluster_centroids = model_kmeans.cluster_centers_


            cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
            colorw=['lime','goldenrod','hotpink','greenyellow','cyan']
            plt.figure(figsize=(17,12))
            for cluster_id, color in cluster_colors.items():
                cluster_data = features[features['cluster'] == cluster_id]
                plt.scatter(cluster_data['market_cap'], cluster_data['volume'], label=f'Cluster {cluster_id+1}', c=color,s=75,alpha=0.5)
                plt.scatter(cluster_centroids[cluster_id, 0], cluster_centroids[cluster_id, 1], marker='*',s=120, c=colorw[cluster_id], label=f'Centroid {cluster_id+1}={cluster_centroids[cluster_id]}',alpha=1)

            plt.title('Coin Market Cap vs. Volume with K-Means Clustering')
            plt.xlabel('Market Cap')
            plt.ylabel('Volume')
            plt.legend()
            plt.grid(True)


            # end code

            st.subheader("First we can see the centeroid for each cluster")
            cluster_names = [f"Cluster {i + 1}" for i in range(len(cluster_centroids))]
            df = pd.DataFrame(cluster_centroids, columns=["Centroid 1", "Centroid 2"], index=cluster_names)
            st.table(df)

            st.subheader("Now to visualize the cluster's with scatter plot :")
            st.pyplot(plt)

    if part_choise == "Part 2":

            st.subheader("In this part we are going to be model the Kmean with the k parameter ranging from 1 to 10 :")

            # code :

            inertias = []
            sil_score=[]
            model_kmeans = KMeans(n_clusters=1, random_state=42)
            features['cluster'] = model_kmeans.fit_predict(features)

            cluster_centroids = model_kmeans.cluster_centers_
            st.write('the cluster centroids will be in \n', cluster_centroids)
            cluster_colors = {0: 'red'}
            colorw=['lime']
            plt.figure(figsize=(17,12))
            for cluster_id, color in cluster_colors.items():
                cluster_data = features[features['cluster'] == cluster_id]
                fit_k=model_kmeans.fit(features)
                inertias.append(fit_k.inertia_)
                plt.scatter(cluster_data['market_cap'], cluster_data['volume'], label=f'Cluster {cluster_id+1}', c=color,s=75,alpha=0.5)
                plt.scatter(cluster_centroids[cluster_id, 0], cluster_centroids[cluster_id, 1], marker='*',s=120, c=colorw[cluster_id], label=f'Centroid {cluster_id+1}={cluster_centroids[cluster_id]}',alpha=1)

            plt.title('Coin Market Cap vs. Volume with K-Means Clustering')
            plt.xlabel('Market Cap')
            plt.ylabel('Volume')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange', 5:'yellow', 6:'cyan', 7:'black', 8:'brown', 9:'gray'}
            colorw=['k','y','r','cyan','m','hotpink','sienna','tan','b','g']

            for K in range(1,10):
                model_kmeans = KMeans(n_clusters=K+1, random_state=42)
                features['cluster'] = model_kmeans.fit_predict(features)

                cluster_centroids = model_kmeans.cluster_centers_
                st.write(f'the cluster centroids  for k = {K+1} will be in \n', cluster_centroids)
                fit_k=model_kmeans.fit(features)
                inertias.append(fit_k.inertia_)
                sil_score.append(silhouette_score(features,fit_k.labels_,metric="euclidean",random_state=200))
                st.write("Silhouette score for k(clusters) = "+str(K+1)+" is "+str(silhouette_score(features,fit_k.labels_,metric="euclidean",random_state=200)))



                plt.figure(figsize=(17, 12))
                for cluster_id, color in cluster_colors.items():
                    if cluster_id > K:
                        break
                    cluster_data = features[features['cluster'] == cluster_id]
                    plt.scatter(cluster_data['market_cap'], cluster_data['volume'], label=f'Cluster {cluster_id+1}', c=color,alpha=0.65,s=75)
                    plt.scatter(cluster_centroids[cluster_id, 0], cluster_centroids[cluster_id, 1], marker='*', c=colorw[cluster_id], s=200, label=f'Centroid {cluster_id+1}={cluster_centroids[cluster_id]}')

                plt.title('Coin Market Cap vs. Volume with K-Means Clustering')
                plt.xlabel('Market Cap')
                plt.ylabel('Volume')
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)
            plt.figure(figsize=(17, 12))
            plt.plot(range(1,11),inertias,'bx-')
            plt.xlabel('value of k')
            plt.ylabel('inertia')
            plt.title('elbow method using inertia')
            st.pyplot(plt)
            sil_centers = pd.DataFrame({'Clusters' : range(2,11), 'Sil Score' : sil_score})
            st.write(sil_centers)
            st.image(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_1\Code\sil.png", caption="Sil Score of Clusters", use_column_width=True)         

            # end code

    if part_choise == "Part 3":
        st.header("In this part we are going to use DBScan to make model with 5 cluster's that has meaningfull info (based on 2 feature's (them being market cap and volume))")
        st.subheader("We are going to be experimenting on the parameters (number of neighbers and alpha) : ")
        
        st.header("First lets see the number of neighbers = 10")

        st.subheader("Some general information :")


        # code 
               
        n_neighbor=10
        neighbor,neighbor_index=NearestNeighbors(n_neighbors=n_neighbor).fit(data[['market_cap','volume']]).kneighbors()
        n=neighbor[:,n_neighbor-1]
        st.write('len:',len(n))
        st.write('max',max(n))
        st.write('min',min(n))
        sort_neighbor=np.sort(n)
        plt.plot(sort_neighbor)
        k_l=KneeLocator(np.arange(len(n)),sort_neighbor,curve='convex')
        st.write('knee',k_l.knee,'\nelbow',k_l.elbow)
        

        # end code


        st.subheader("The sorted neighbor plot :")
        st.pyplot(plt)
        st.subheader("The Knee point plot :")
        st.image(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_1\Code\Knee.png", caption = "Knee point plot", use_column_width=True)

        # code :

        dbscan=DBSCAN(eps=16921911111.611433,min_samples=n_neighbor)
        dbscan.fit(data[['market_cap','volume']])
        plt.figure(figsize=(12, 6))
        plt.scatter(data['market_cap'], data['volume'], c=dbscan.labels_)
        plt.grid(True)
        plt.xlabel('MarketCap')
        plt.ylabel('Volume')
        plt.title('DBSCAN Clustering')


        np.unique(dbscan.labels_,return_counts=True)

        noise=list(dbscan.labels_).count(-1)
        noise = noise*100/len(dbscan.labels_)

        # end code 

        st.subheader("Market cap vs Volume scatter plot :")
        st.pyplot(plt)

        st.subheader("And the noise would be :")
        st.write(f"Noise: {noise}", key="number_box", format="0")


        st.header("Now to see the number of neighbers = 4")

        st.subheader("Some general information :")


        # code 
               
        n_neighbor=4
        neighbor,neighbor_index=NearestNeighbors(n_neighbors=n_neighbor).fit(data[['market_cap','volume']]).kneighbors()
        n=neighbor[:,n_neighbor-1]
        st.write('len:',len(n))
        st.write('max',max(n))
        st.write('min',min(n))
        sort_neighbor=np.sort(n)
        plt.plot(sort_neighbor)
        k_l=KneeLocator(np.arange(len(n)),sort_neighbor,curve='convex')
        st.write('knee',k_l.knee,'\nelbow',k_l.elbow)
        

        # end code


        st.subheader("The sorted neighbor plot :")
        st.pyplot(plt)
        st.subheader("The Knee point plot :")
        st.image(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_1\Code\Knee_4.png", caption = "Knee point plot", use_column_width=True)

        # code :

        from sklearn.cluster import DBSCAN
        dbscan=DBSCAN(eps=14100000000,min_samples=n_neighbor)#eps=0.2081727,min_samples=8
        dbscan.fit(data[['market_cap','volume']])
        plt.figure(figsize=(12, 6))
        plt.scatter(data['market_cap'], data['volume'], c=dbscan.labels_)
        plt.grid(True)
        plt.xlabel('MarketCap')
        plt.ylabel('Volume')
        plt.title('DBSCAN Clustering')


        np.unique(dbscan.labels_,return_counts=True)

        noise=list(dbscan.labels_).count(-1)
        noise*100/len(dbscan.labels_)

        # end code 

        st.subheader("Market cap vs Volume scatter plot :")
        st.pyplot(plt)

        st.subheader("And the noise would be :")
        st.write(f"Noise: {noise}", key="number_box", format="0")

        st.header("For this part we are going to alter the data to reduce the noise (and we are going to use number of neighbers = 10)")

        st.subheader("Some general information :")


        # code 
               
        G=data.loc[data['volume']<1*1e11]
        n_neighbor=10
        neighbor,neighbor_index=NearestNeighbors(n_neighbors=n_neighbor).fit(G[['market_cap','volume']]).kneighbors()
        n=neighbor[:,n_neighbor-1]
        st.write('len:',len(n))
        st.write('max',max(n))
        st.write('min',min(n))
        sort_neighbor=np.sort(n)
        plt.plot(sort_neighbor)
        k_l=KneeLocator(np.arange(len(n)),sort_neighbor,curve='convex')
        st.write('knee',k_l.knee,'\nelbow',k_l.elbow)
        

        # end code


        st.subheader("The sorted neighbor plot :")
        st.pyplot(plt)
        st.subheader("The Knee point plot :")
        st.image(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_1\Code\Knee_last.png", caption = "Knee point plot", use_column_width=True)

        # code :

        dbscan=DBSCAN(eps=16899999999,min_samples=n_neighbor)#eps=0.2081727,min_samples=8
        dbscan.fit(G[['market_cap','volume']])
        plt.figure(figsize=(12, 6))
        plt.scatter(G['market_cap'], G['volume'], c=dbscan.labels_)
        plt.grid(True)
        plt.xlabel('MarketCap')
        plt.ylabel('Volume')
        plt.title('DBSCAN Clustering')


        np.unique(dbscan.labels_,return_counts=True)

        noise=list(dbscan.labels_).count(-1)
        noise = noise*100/len(dbscan.labels_)

        # end code 

        st.subheader("Market cap vs Volume scatter plot :")
        st.pyplot(plt)

        st.subheader("And the noise would be :")
        st.write(f"Noise: {noise}", key="number_box", format="0")



if report_choice == "Report 2":


    st.header("Report 2")
    data = pd.read_excel(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_2\Data\coins_data.xlsx')

    st.subheader("Which part are we going for ?")
    part_choise = st.selectbox("", ["Part 1", "Part 2", "Part 3", "Part 4"])



    if part_choise == "Part 1":
        st.header("In this part we use hierarchical clustering with 2 cluster's and 2 features.")
        st.write("First we should take a look at the Dendogram we drew for this part :")


        # Code :
    
        coin_data = pd.read_excel(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_2\Data\coins_data.xlsx')
        features = coin_data[['MarketCap', 'Volume']]

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        from scipy.cluster.hierarchy import dendrogram, linkage
        import matplotlib.pyplot as plt

        Z = linkage(features, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(Z, orientation='top')
        plt.title('Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')

        # end code

        st.pyplot(plt)


        st.header("First a general explanation on the Dendogram:")
        st.write("""Ok so the dendogram graph is shown above, but what does it mean ?
        the dendogram is generaly going tp show use the data as hierarchical clusters, meaning at the start each observation is a point in the x_axis 
        and as we come up in the dendogram, these data points join togheter to form the clusters at their respective levels.
        in this project we are given the number of clusters, but for completion sake we are going to try to figure out the optimal amount of clusters.
        the manner in which we choose the number of clusters in a dendogram is rather instinctive, but there is a way to start quesing.
        we have to draw a horizantal line through the plot, and however many colisions we have with the lines in the plot, are the number of clusters.
        in this case i think we are better off having 3 clusters. (after the second part we will discuss this fully)
        """)


        # code :

        n_clusters = 2
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)


        cluster_0 = coin_data[coin_data['Cluster'] == 0]
        cluster_1 = coin_data[coin_data['Cluster'] == 1]

        cluster_0_names = cluster_0['Symbol'].tolist()
        cluster_1_names = cluster_1['Symbol'].tolist()


        # end code

        st.subheader("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")


        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""This cluster includes the cryptocurrencies "USDT" and "BTC." 
                    These Cryptocurrencies stand out as the cryptocurrencies with the highest market capitalization and trading volume among the 20 cryptocurrencies.""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""Cluster 2 is the larger cluster, containing a wide range of cryptocurrencies, including "LEO," "UNI," "WBTC," "AVAX," "DAI," "SHIB," "LTC," "TRX," "DOT," "MATIC," "SOL," "DOGE," "ADA," "BUSD," "XRP," "USDC," "BNB," and "ETH."
                    These cryptocurrencies generally have lower market capitalization and trading volume compared to those in Cluster 0.""")
        
        st.subheader("Generally speaking")
        st.write("""The model has grouped cryptocurrencies based on their market cap and volume. 
                    As expected it has separated Bitcoin and USDT, which are notably different from the rest of the cryptocurrencies due to their much higher market capitalization and trading volume.
                    If we were to need more meaningfull clusters or different groupings, we could experiment with different clustering techniques, features, or adjust the number of clusters based on our specific goal.""")
        

    if part_choise == "Part 2":
        st.header("In this part we use hierarchical clustering with 2 cluster's but with 3 features.")
        st.write("First we should take a look at the Dendogram we drew for this part :")

        # Code :
    
        coin_data = pd.read_excel(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_2\Data\coins_data.xlsx')

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        coin_data['ProofType_Label'] = label_encoder.fit_transform(coin_data['ProofType'])

        features = coin_data[['MarketCap', 'Volume', 'ProofType_Label']]

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        n_clusters = 2
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)

        from scipy.cluster.hierarchy import dendrogram, linkage
        import matplotlib.pyplot as plt

        Z = linkage(features, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(Z, orientation='top')
        plt.title('Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')

        # end code

        st.pyplot(plt)

        # code :

        n_clusters = 2
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)


        cluster_0 = coin_data[coin_data['Cluster'] == 0]
        cluster_1 = coin_data[coin_data['Cluster'] == 1]

        cluster_0_names = cluster_0['Symbol'].tolist()
        cluster_1_names = cluster_1['Symbol'].tolist()


        # end code

        st.subheader("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")



        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""This cluster primarily consists of cryptocurrencies that utilize Proof of Stake (PoS) and some stablecoin-based cryptocurrencies.
                    Notably, these cryptocurrencies operate on various blockchain networks, such as Ethereum, Tron, Solana, and others.
                    The "MarketCap" and "Volume" vary significantly within this cluster, indicating diversity in terms of market capitalization and trading volume.""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""Cluster 2 is primarily represented by cryptocurrencies that use Proof of Work (PoW) as their consensus mechanism, including Bitcoin (BTC) and USDT.
                    This cluster is less diverse in terms of the "ProofType," with a dominance of PoW-based cryptocurrencies.
                    The "MarketCap" and "Volume" values in this cluster also exhibit variation, with Bitcoin and USDT standing out as 2 cryptocurrencies with a high market capitalization.""")
        
        st.subheader("So why are they the same as before ?")
        st.write("""While keeping the number of clusters constant at two, including the "ProofType" feature has resulted in more refined clusters. 
                    Cluster 1 consists of a mix of PoS-based and stablecoin-based cryptocurrencies, while Cluster 2 predominantly comprises PoW-based cryptocurrencies. 
                    This analysis provides insights into how the "ProofType" feature can influence the clustering results and offers a way to distinguish cryptocurrencies based on their consensus mechanisms within a two-cluster framework.""")


        st.header("So what conclusions can we draw from these results ?")
        st.write("""Seeing as how the only thing different in both parts is the number of features (although for the second part choosing 3 clusters could potentially be beneficial), we can confidently say the choice of adding a feature can make the results more reliable.
                    But as the clusters have remained exactly the same, we can conclude only changing the number of features and not the number of clusters could hurt the results.
                    But if we were to change the number of clusters (as demonstrated below), we can take advantage of both of the changes.
                    In conclusion, the second part of the project demonstrated that incorporating additional relevant features can lead to more informative and actionable clustering results, making it a preferred approach when analyzing the cryptocurrency market. 
                    The choice of features is essential in the world of cryptocurrencies.""")




    if part_choise == "Part 3":
        st.header("In this part we use hierarchical clustering with 3 features but with 3 clusters as well (experimenting on the number of clusters)")
        st.write("First we should take a look at the Dendogram we drew for this part :")

        # Code :
    
        coin_data = pd.read_excel(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_2\Data\coins_data.xlsx')

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        coin_data['ProofType_Label'] = label_encoder.fit_transform(coin_data['ProofType'])

        features = coin_data[['MarketCap', 'Volume', 'ProofType_Label']]

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        n_clusters = 3
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)

        from scipy.cluster.hierarchy import dendrogram, linkage
        import matplotlib.pyplot as plt

        Z = linkage(features, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(Z, orientation='top')
        plt.title('Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')

        # end code

        st.pyplot(plt)

        # code :

        n_clusters = 3
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)

        cluster_0 = coin_data[coin_data['Cluster'] == 0]
        cluster_1 = coin_data[coin_data['Cluster'] == 1]
        cluster_2 = coin_data[coin_data['Cluster'] == 2]

        cluster_0_names = cluster_0['Symbol'].tolist()
        cluster_1_names = cluster_1['Symbol'].tolist()
        cluster_2_names = cluster_2['Symbol'].tolist()


        # end code

        st.subheader("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the third one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")



        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""Both 'USDT' and 'BTC' have high market capitalizations.
                    These two cryptocurrencies share the same Proof Type of 'stablecoin.
                    This cluster primarily includes stablecoins, with 'USDT' and 'BTC' having significant market capitalizations. 
                    Stablecoins are designed to maintain a stable value and are often used for trading and transferring value.""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""This cluster appears to include a diverse set of cryptocurrencies with varying market capitalizations and trading volumes.
                    The 'ProofType_Label' values in this cluster differ from 'PoS' to 'PoW' and 'PoH,' suggesting a mix of Proof Types.
                    Cluster 2 represents a diverse portfolio of cryptocurrencies, including well-established ones like 'ETH,' 'ADA,' and 'BNB,' along with a mix of others. 
                    The cryptocurrencies in this cluster have various Proof Types, indicating diversity in how they secure their networks.""")
        
        st.subheader("What about cluster 3 ?")
        st.write("""These cryptocurrencies are associated with the 'stablecoin' Proof Type.
                    'DAI' and 'BUSD' are stablecoins on the Ethereum network, while 'XRP' and 'USDC' are also stablecoins with different Proof Types.
                    This cluster is composed of cryptocurrencies with the 'stablecoin' Proof Type. 
                    'DAI' and 'BUSD' are stablecoins on the Ethereum network, while 'XRP' and 'USDC' are also stablecoins with different Proof Types.""")



        st.subheader("So why are they the same as before ?")
        st.write("""It's essential to consider the specific characteristics of the cryptocurrencies within each cluster when making investment or analysis decisions. 
                    The choice of three clusters allows for a more meaningfull view of the cryptocurrency market, considering market capitalization, trading volume, and the type of network security (Proof Type). 
                    As always, it's important to conduct further research and analysis to make informed decisions about cryptocurrency investments.""")




    if part_choise == "Part 4":
        st.header("In this part we use hierarchical clustering with 2 clusters but with more features (we want to see the effect of the number of features here)")
        st.write("First we should take a look at the Dendogram we drew for this part as well :")

        # Code :
    
        coin_data = pd.read_excel(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_2\Data\coins_data.xlsx')

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        coin_data['ProofType_Label'] = label_encoder.fit_transform(coin_data['ProofType'])

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        coin_data['Network_Label'] = label_encoder.fit_transform(coin_data['Network'])


        features = coin_data[['MarketCap', 'Volume', 'ProofType_Label', 'Network_Label']]

        scaler = StandardScaler()
        features = scaler.fit_transform(features)


        from scipy.cluster.hierarchy import dendrogram, linkage
        import matplotlib.pyplot as plt

        Z = linkage(features, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(Z, orientation='top')
        plt.title('Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')

        # end code

        st.pyplot(plt)
        st.write("As we can see, the vertical distance and the number of colors used in the dendogram indicates that the desired number of clusters is 2")

        # code :

        n_clusters = 2
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        coin_data['Cluster'] = hc.fit_predict(features)

        cluster_0 = coin_data[coin_data['Cluster'] == 0]
        cluster_1 = coin_data[coin_data['Cluster'] == 1]

        cluster_0_names = cluster_0['Symbol'].tolist()
        cluster_1_names = cluster_1['Symbol'].tolist()

        print("Cryptocurrencies in Cluster 0:")
        print(cluster_0_names)

        print("\nCryptocurrencies in Cluster 1:")
        print(cluster_1_names)


        # end code

        st.subheader("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 5].write(f"- {symbol}")


        st.header("Conclusion :")
        st.subheader("Cluster Composition:")
        st.write("""Cluster 1 is more diverse in terms of the cryptocurrencies it includes, while Cluster 2 comprises two of the most significant cryptocurrencies in the market, 'USDT' and 'BTC.'
""")

        st.subheader("Diversity in Features:")
        st.write("""The four features used in clustering (market capitalization, volume, ProofType_Label, Network_Label) capture various aspects of cryptocurrencies, including their technology (ProofType) and underlying blockchain networks (Network_Label).
""")
        

        st.subheader("Risk and Stability:")
        st.write("""Cluster 2, with 'USDT' and 'BTC,' represents assets known for their stability and low volatility. These are often seen as safe havens in the crypto market.
""")

        st.subheader("Cluster 1 Diversity: ")
        st.write("""Cryptocurrencies in Cluster 1 exhibit a wide range of characteristics. Investors seeking diversification or those interested in exploring different types of cryptocurrencies may find this cluster appealing.
""")
        
        st.subheader("Investment Strategy: ")
        st.write("""Your choice between Cluster 1 and Cluster 2 may depend on your investment strategy. Cluster 2 is generally considered less risky, while Cluster 1 offers more options for potential growth but with varying risk levels.
""")
        
        st.subheader("Analysis Continuation: ")
        st.write("""Beyond clustering, you should analyze the individual characteristics and performance of the cryptocurrencies within each cluster to make informed investment decisions.
""")







if report_choice == "Report 3":

    # code :

    train = pd.read_csv(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Data\train.csv')
    teste = pd.read_csv(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Data\teste.csv')
    testf = pd.read_csv(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Data\testf.csv')

    train.drop(columns = ['Date'], inplace = True)
    teste.drop(columns = ['Date'], inplace = True)
    testf.drop(columns = ['Date'], inplace = True)

    Xtrain = train.drop(columns=['Target_XMR'])
    ytrain = train['Target_XMR']
    Xeval = teste.drop(columns=['Target_XMR'])
    yeval = teste['Target_XMR']
    Xtest = testf.drop(columns=['Target_XMR'])
    ytest = testf['Target_XMR']

    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    Xtrain = pca.fit_transform(Xtrain)
    Xtest = pca.transform(Xtest)
    Xeval = pca.transform(Xeval)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()


    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    Xeval = scaler.transform(Xeval)

    # end code


    st.header("Report 3")

    st.subheader("Choose a Dataset to Display:")
    dataset_choice = st.selectbox("Select a Dataset", ["Train", "Teste", "Testf"])

    if dataset_choice == "Train":
        data = pd.read_csv(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Data\train.csv')
    elif dataset_choice == "Teste":
        data = pd.read_csv(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Data\teste.csv')
    elif dataset_choice == "Testf":
        data = pd.read_csv(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Data\testf.csv')

    st.write(f"Displaying {dataset_choice} Dataset:")
    st.dataframe(data)

    st.write("First we prep the data (check for imbalancement and decomposite the data and reduce dimensionallity and standardizing it)")


    st.header("We created 3 models for this part compared them and chose the best one.")

    model_choice = st.selectbox("Which model would you like to see ?", ["Model 1", "Model 2", "Model 3"])


    if model_choice == "Model 1":


        st.header("For the first model we used the KNN method :")

        # code 

        from sklearn.neighbors import KNeighborsClassifier
        model1 = KNeighborsClassifier(n_neighbors=3)

        model1.fit(Xtrain, ytrain)
        ypred = model1.predict(Xeval)
        score = f1_score(yeval, ypred)

        fig , ax = plt.subplots(figsize = (8, 2))

        plt.scatter(range(1, Xeval.shape[0]+1), yeval, label = 'answer')

        plt.scatter(range(1, Xeval.shape[0]+1), ypred, label = 'predictions', alpha=0.7)

        plt.legend(loc = 'center')

        # end code

        st.pyplot(plt)
        F_1_Score = score
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("The F1 score for this model is about 0.55, and so this method is not so appealing.")


    elif model_choice == "Model 2":



        # code :

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=2000, min_samples_split=20, random_state=1, bootstrap=True, oob_score=True, max_samples=1000)

        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xeval)

        score = f1_score(yeval, ypred)

        fig , ax = plt.subplots(figsize = (8, 2))

        plt.scatter(range(1, Xeval.shape[0]+1), yeval, label = 'answer')

        plt.scatter(range(1, Xeval.shape[0]+1), ypred, label = 'predictions', alpha=0.7)

        plt.legend(loc = 'center')

        ypreds = model.predict(Xtest)

        score_1 = f1_score(ytest, ypreds)

        # end code
        
        st.header("For the Second model we used the Random Forest method :")
        st.pyplot(plt)
        F_1_Score = score
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("The F1 score for this method is about 0.68, and compared to the previous one and in general it is accaptable for now.")


        st.write("So far we have used the eval data to find the F_1 score and to compare, but now that we have chosen the method we are gonna use, we might aswell use the actuall test data and see the F1 score for it.")

        # code :

        fig , ax = plt.subplots(figsize = (8, 2))

        plt.scatter(range(1, Xtest.shape[0]+1), ytest, label = 'answer')

        plt.scatter(range(1, Xtest.shape[0]+1), ypreds, label = 'predictions', alpha=0.7)

        plt.legend(loc = 'center')


        # end code

        st.pyplot(plt)
        F_1_Score = score_1
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("We can see that the F1 score has gone up compered to before, this could be because of the fact that our two sets are located in different trend intervals and therefore have different distribution of 0s and 1s.")

        st.header("For ADHD sake, we are going to implement a backtracking system :")

        # code :

        pf = cf.copy()
        pf.drop(columns=['Date'], inplace = True)

        def backtest(data, start = 2114, step = 1):
            all_predictions = []
            for i in range(start, data.shape[0], step):
                train = data.iloc[0:i].copy()
                test = data.iloc[i:(i+step)].copy()
                pca = PCA(n_components=3)
                Xtrain = train.drop(columns=['Target_XMR'])
                Xtrain = pca.fit_transform(Xtrain)
                ytrains = train['Target_XMR'] 
                Xtest = test.drop(columns=['Target_XMR'])
                Xtest = pca.transform(Xtest)
                scaler = StandardScaler()
                Xtrain = scaler.fit_transform(Xtrain)
                Xtest = scaler.transform(Xtest)
                models = RandomForestClassifier(n_estimators=2000, min_samples_split=20, random_state=1, bootstrap=True, oob_score=True, max_samples=1000)
                models.fit(Xtrain, ytrains)
                predictions = models.predict(Xtest)
                all_predictions.append(predictions)
            return all_predictions

        preds = backtest(pf)
        y = np.array(preds).flatten()
        yhat = ytest
        score = f1_score(yhat, y)


        fig , ax = plt.subplots(figsize = (8, 2))

        plt.scatter(range(1, Xtest.shape[0]+1), yhat, label = 'answer')

        plt.scatter(range(1, Xtest.shape[0]+1), y, label = 'predictions', alpha=0.7)

        plt.legend(loc = 'center')

        # end code 

        st.pyplot(plt) 
        F_1_Score = score
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("""So, if we use a 30 day interval as trend, we can find the next trend with about 74% accuracy on both 1s and 0s, but if we try to do this mid-trends and one by one like our backtesting model; we will see a little bit of setback because the model is not capabale of finding itself's location in the respective trend.
                    Therefore our backtesting system's accuracy on both 1s and 0s would be about 67%.""")


    elif model_choice == "Model 3":


        # code :

        Xtrain = train.drop(columns=['Target_XMR'])
        ytrain = train['Target_XMR']

        Xeval = teste.drop(columns=['Target_XMR'])
        yeval = teste['Target_XMR']

        Xtest = testf.drop(columns=['Target_XMR'])
        ytest = testf['Target_XMR']

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.transform(Xtest)
        Xeval = scaler.transform(Xeval)

        from sklearn.ensemble import AdaBoostClassifier
        model3 = AdaBoostClassifier(n_estimators=350)

        model3.fit(Xtrain, ytrain)
        ypred = model3.predict(Xeval)

        score = f1_score(yeval, ypred)

        fig , ax = plt.subplots(figsize = (8, 2))

        plt.scatter(range(1, Xeval.shape[0]+1), yeval, label = 'answer')

        plt.scatter(range(1, Xeval.shape[0]+1), ypred, label = 'predictions', alpha=0.7)

        plt.legend(loc = 'center')

        # end code
        
        st.header("As we saw, the f1 score for the randoom forest is acceptable but we can use more complex models such as AdaBoostClassifier :")
        st.pyplot(plt)
        F_1_Score = score
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("Not very apealing, but we might as well use out test data aswell.")

        # code :

        ypreds = model3.predict(Xtest)
        score = f1_score(ytest, ypreds)

        # end score

        st.pyplot(plt)
        F_1_Score = score
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")

        # code 

        pf = cf.copy()
        pf.drop(columns=['Date'], inplace = True)
        def backtest(data, start = 2114, step = 1):
            all_predictions = []
            for i in range(start, data.shape[0], step):
                train = data.iloc[0:i].copy()
                test = data.iloc[i:(i+step)].copy()
                Xtrain = train.drop(columns=['Target_XMR'])
                ytrains = train['Target_XMR'] 
                Xtest = test.drop(columns=['Target_XMR'])
                scaler = StandardScaler()
                Xtrain = scaler.fit_transform(Xtrain)
                Xtest = scaler.transform(Xtest)
                models = AdaBoostClassifier(n_estimators=350)
                models.fit(Xtrain, ytrains)
                predictions = model3.predict(Xtest)
                all_predictions.append(predictions)
            return all_predictions

        preds = backtest(pf)
        y = np.array(preds).flatten()
        yhat = ytest
        score = f1_score(yhat, y)

        fig , ax = plt.subplots(figsize = (8, 2))

        plt.scatter(range(1, Xtest.shape[0]+1), yhat, label = 'answer')

        plt.scatter(range(1, Xtest.shape[0]+1), y, label = 'predictions', alpha=0.7)

        plt.legend(loc = 'center')

        # end code


        st.write("Let's see how the back testing works :")
        st.pyplot(plt)
        F_1_Score = score
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("""As you can see, AdaBoost is more consistent with Backtesting which is a better measure for our F1 Score because of a better distribution of 1s and 0s.
                    Finally we can say our RandomForest is better in a full trend finding way with a F1-score of 74% and has a pretty good backtesting F1-score of 67% approximately.
                    Also our AdaBoost model is slightly better in backtesting with more than 68% in F1-score.""")





