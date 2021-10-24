
\subsection{Data}

\textbf{Data Folder} contains two sub-folders: \textbf{Synthetic Data Folder} includes training data and testing data generated; \textbf{RCT Data Folder} contains real-world RCT data which has been split into training and testing.

\textbf{Synthetic Data Folder} contains generated simulation data files with two types (training and testing) and four uncertainty levels. These data files are used for offline evaluation mentioned in section 5.1. These data files were generated following the description in section A.2, with different uncertainty weight levels. The unobserved variable was specifically discarded from training data to satisfy the "unobserved" condition.


\begin{enumerate}

\item \textit{Synthetic Data} - Each training sample was attributed with the following:
\begin{itemize}
\item  ID - Column[0], data unique ID.
\item  Heterogeneous Variables - Column[1-4].
\item  Treatment Variable - Column[5], with four different level of treatments.
\item  Outcome Variables - Column[6-7], Value and Cost under BTS problem setting.
\end{itemize}

\end{enumerate}

\textbf{RCT Data Folder} contains web-scale RCT data collected from a video streaming platform which was used as training instances. The dataset records the users' campaign engagement duration (i.e., ``outcome'') in seven randomly enrolled incentive groups, each offered bonuses at different levels (i.e., ``multi-treatment''). In the experiments of Section 5.2, our dataset consists of over 100 K app visit instances. 

Due to the privacy nature of the data, currently we are not able to disclose all the actual data used in our experiments until publication but the practitioner can use the code provided to run this algorithm on other dataset. 
However, a small part (about 2000 samples) of real-world data are provided in \textbf{Data Folder} just for running the code, \textbf{NOT FOR REPRODUCE THE RESULT} in Section 5.2, and \textbf{ALL THE 2000 SAMPLES HAVE BEEN Encrypted}. As stated in the paper, the complete dataset used in offline test will be released upon publication.
