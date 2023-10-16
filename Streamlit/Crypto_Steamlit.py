import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


st.title("Cryptocurrency Analysis")

report_choice = st.sidebar.selectbox("Select a Report", ["Report 1", "Report 2", "Report 3"])

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

        st.write("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(1)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 1].write(f"- {symbol}")


        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""This cluster contains the majority of the cryptocurrencies, including LEO, UNI, WBTC, AVAX, and others.
                    These cryptocurrencies generally have lower market capitalization and trading volume compared to those in Cluster 2.
                    Proof types in this cluster vary between PoS, PoW, stablecoin, PoH, and RPCA, indicating diversity in the types of cryptocurrencies.""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""This cluster contains a single cryptocurrency, which is Bitcoin (BTC).
                    Bitcoin stands out as the cryptocurrency with the highest market capitalization and trading volume among the 20 cryptocurrencies.""")
        
        st.subheader("Generally speaking")
        st.write("""The model has grouped cryptocurrencies based on their market cap and volume. 
                    As expected it has separated Bitcoin, which is notably different from the rest of the cryptocurrencies due to its much higher market capitalization and trading volume.
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

        st.write("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(1)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 1].write(f"- {symbol}")



        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""This cluster primarily consists of cryptocurrencies that utilize Proof of Stake (PoS) and some stablecoin-based cryptocurrencies.
                    Notably, these cryptocurrencies operate on various blockchain networks, such as Ethereum, Tron, Solana, and others.
                    The "MarketCap" and "Volume" vary significantly within this cluster, indicating diversity in terms of market capitalization and trading volume.""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""Cluster 1 is primarily represented by cryptocurrencies that use Proof of Work (PoW) as their consensus mechanism, including Bitcoin (BTC).
                    This cluster is less diverse in terms of the "ProofType," with a dominance of PoW-based cryptocurrencies.
                    The "MarketCap" and "Volume" values in this cluster also exhibit variation, with Bitcoin standing out as a cryptocurrency with a high market capitalization.""")
        
        st.subheader("So why are they the same as before ?")
        st.write("""While keeping the number of clusters constant at two, including the "ProofType" feature has resulted in more refined clusters. 
                    Cluster 0 consists of a mix of PoS-based and stablecoin-based cryptocurrencies, while Cluster 1 predominantly comprises PoW-based cryptocurrencies. 
                    This analysis provides insights into how the "ProofType" feature can influence the clustering results and offers a way to distinguish cryptocurrencies based on their consensus mechanisms within a two-cluster framework.""")


        st.header("So what conclusions can we draw from these results ?")
        st.write("""Seeing as how the only thing different in both parts is the number of features (although for the second part choosing 3 clusters could potentially be beneficial), we can confidently say the choice of adding a feature can make the results more reliable.""")
        st.write("""But as the clusters have remained exactly the same, we can conclude only changing the number of features and not the number of clusters could hurt the results.""")
        st.write("""But if we were to change the number of clusters (as demonstrated below), can take advantage of both of the changes.""")
        st.write("""In conclusion, the second part of the project demonstrated that incorporating additional relevant features can lead to more informative and actionable clustering results, making it a preferred approach when analyzing the cryptocurrency market. """)
        st.write("""The choice of features is essential in the world of cryptocurrencies.""")




    if part_choise == "Part 3":
        st.header("In this part we use hierarchical clustering with 3 features but with 3 clusters as well (experomenting on the number of clusters)")
        st.write("First we should take a look at the Dendogram we drew for this part :")

        # Code :
    
        coin_data = pd.read_excel(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_2\Data\coins_data.xlsx')

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        coin_data['ProofType_Label'] = label_encoder.fit_transform(coin_data['ProofType'])

        features = coin_data[['MarketCap', 'Volume', 'ProofType_Label']]

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

        st.write("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 1].write(f"- {symbol}")
        st.subheader("For the third one :")
        symbol_columns = st.columns(1)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 1].write(f"- {symbol}")



        st.subheader("So what does a coin being in the first cluster mean ?")
        st.write("""In Cluster 1, we have a diverse group of cryptocurrencies, including LEO, UNI, WBTC, AVAX, DAI, SHIB, LTC, TRX, DOT, MATIC, SOL, DOGE, ADA, BUSD, XRP, USDC, BNB, and USDT.
                    This cluster represents a wide range of cryptocurrencies, both in terms of market capitalization and volume.
                    The majority of cryptocurrencies in this cluster are well-established and widely traded.
                    It includes a mix of proof-of-stake (PoS), proof-of-work (PoW), and stablecoin-based cryptocurrencies.
                    Ethereum is the most common network among them, indicating that Ethereum hosts a diverse range of projects.""")
        
        st.subheader("What about cluster 2 ?")
        st.write("""Cluster 2 contains a single cryptocurrency, BTC (Bitcoin).
                    Bitcoin is the most well-known and valuable cryptocurrency globally, with a high market capitalization.
                    It utilizes the proof-of-work (PoW) consensus mechanism.
                    Bitcoin operates on its blockchain network, distinct from other cryptocurrencies in this dataset.""")
        
        st.subheader("So why are they the same as before ?")
        st.write("""Cluster 3 also includes a single cryptocurrency, ETH (Ethereum).
                    Ethereum, like Bitcoin, is a major player in the cryptocurrency space.
                    Ethereum is known for its smart contract capabilities and decentralized applications (DApps).
                    It uses the proof-of-stake (PoS) consensus mechanism and operates on the Ethereum blockchain network.""")


        st.header("Effects of 3 clusters :")
        st.write("""This cluster primarily consists of cryptocurrencies that utilize Proof of Stake (PoS) and some stablecoin-based cryptocurrencies.
                    Notably, these cryptocurrencies operate on various blockchain networks, such as Ethereum, Tron, Solana, and others.
                    The "MarketCap" and "Volume" vary significantly within this cluster, indicating diversity in terms of market capitalization and trading volume.
                    Creating three clusters allows for a more detailed classification of cryptocurrencies compared to two clusters. It separates Bitcoin and Ethereum into their own clusters, emphasizing their significance and unique characteristics, but this effect only takes place if we change both the number of clusters **AND** the number of features.""")
        
        st.header("Bitcoin and Ethereum Dominance")
        st.write("""Bitcoin (BTC) and Ethereum (ETH) are distinct from the rest of the cryptocurrencies due to their significant market capitalization and unique features.""")

        st.header("Diverse Cluster 1 :")
        st.write("""Cluster 1 is diverse, comprising a wide range of cryptocurrencies, including stablecoins, PoS, and PoW-based assets. The common use of the Ethereum network highlights Ethereum's versatility as a platform for various projects.""")


        st.header("Investment Strategy :")
        st.write("""Investors and stakeholders might consider a different investment strategy for cryptocurrencies in each cluster. For instance, Cluster 0 could be considered for diversification, while Bitcoin and Ethereum might be seen as long-term store-of-value assets.""")



    if part_choise == "Part 4":
        st.header("In this part we use hierarchical clustering with 2 clusters but with more features (we want to see the effect of the number of features here)")
        st.write("First we should take a look at the Dendogram we drew for this part per usuall :")

        # Code :
    
        coin_data = pd.read_excel(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_2\Data\coins_data.xlsx')

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        coin_data['ProofType_Label'] = label_encoder.fit_transform(coin_data['ProofType'])

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        coin_data['Network_Label'] = label_encoder.fit_transform(coin_data['Network'])


        features = coin_data[['MarketCap', 'Volume', 'ProofType_Label', 'Network_Label']]


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
        st.write("As we can see, the vertical distance is greaterin the blue line indicating that the desired number of clusters is 2")

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

        st.write("For comparison sake we need to know what is inside each cluster :")
        st.subheader("For the first one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_0_names):
            symbol_columns[i % 5].write(f"- {symbol}")
        st.subheader("For the second one :")
        symbol_columns = st.columns(5)
        for i, symbol in enumerate(cluster_1_names):
            symbol_columns[i % 1].write(f"- {symbol}")


        st.header("Conclusion :")
        st.write("""As we can see, adding the "Network" feature helped in creating more interpretable and meaningful clusters. 
                    The "Network" feature represents the underlying blockchain network on which a cryptocurrency operates. 
                    When we initially clustered the data using only the "MarketCap," "Volume," and "ProofType_Label" features, the clusters did not show distinct patterns. However, after adding the "Network" feature and performing clustering with it, the clusters became more interpretable. 
                    This suggests that the "Network" feature provided valuable information for distinguishing between cryptocurrencies.
                    In regardes to the actuall market, adding the network feature can be helpful in these manner : Blockchain Ecosystem Analysis, Risk Assessment, Use Case Identification, Regulatory Compliance, Market Dynamics, Community and Development.""")






if report_choice == "Report 3":
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

    st.header("We created 3 models for this part compared them and chose the best one.")

    model_choice = st.selectbox("Which model would you like to see ?", ["Model 1", "Model 2", "Model 3"])


    if model_choice == "Model 1":


        st.header("For the first model we used the KNN method :")
        st.image("D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Report\KNN", caption="KNN Accuracy Plot")
        F_1_Score = 0.55
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("The F1 score for this model is about 0.55, and so this method is not so appealing.")


    elif model_choice == "Model 2":

        
        st.header("For the Second model we used the Random Forest method :")
        st.image("D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Report\Decision Tree", caption="Random Forest Accuracy Plot", use_column_width=True)
        F_1_Score = 0.68
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("The F1 score for this method is about 0.68, and compared to the previous one and in general it is accaptable for now.")


        st.write("So far we have used the eval data to find the F_1 score and to compare, but now that we have chosen the method we are gonna use, we might aswell use the actuall test data and see the F1 score for it.")
        st.image("D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Report\Decision Tree test", caption="Random Forest Accuracy Plot (for test data)", use_column_width=True)
        F_1_Score = 0.74
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("We can see that the F1 score has gone up compered to before, this could be because of the fact that our two sets are located in different trend intervals and therefore have different distribution of 0s and 1s.")

        st.header("For ADHD sake, we are going to implement a backtracking system :")
        st.image("D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Report\Decision Tree back tracking", caption="Random Forest Accuracy Plot (back track)", use_column_width=True)
        F_1_Score = 0.71
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("So, if we use a 30 day interval as trend, we can find the next trend with about 74% accuracy on both 1s and 0s, but if we try to do this mid-trends and one by one like our backtesting model; we will see a little bit of setback because the model is not capabale of finding it's location in the respective trend. ")
        st.write("Therefore our backtesting system's accuracy on both 1s and 0s would be about 71%.")


    elif model_choice == "Model 3":


        st.header("As we saw, the f1 score for the randoom forest is acceptable but we can use more complex models such as AdaBoostClassifier :")
        st.image("D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Report\AdaBoostClassifier eval", caption="AdaBoostClassifier Accuracy Plot (for eval data)", use_column_width=True)
        F_1_Score = 0.53
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")
        st.write("Not very apealing, but we might as well use out test data aswell.")

        st.image("D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Report\AdaBoostClassifier test", caption="AdaBoostClassifier Accuracy Plot (for test data)", use_column_width=True)
        F_1_Score = 0.78
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")

        st.write("Let's see how the back testing works :")
        st.image("D:\Sharif University of Tech\Data Sience Boot Camp\Project\Second Phaze\Part_3\Report\AdaBoostClassifier back track", caption="AdaBoostClassifier Accuracy Plot (back track)", use_column_width=True)
        F_1_Score = 0.76
        st.write(f"F_1 Score: {F_1_Score}", key="number_box", format="0")





