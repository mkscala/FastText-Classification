# FastText-Classification

FastText-Classification is an NLP task meant to help identify the class of a given text. 
For this work we employ an open-source dataset from Kaggle website namely BBC News. The dataset consist news for the following classes: Technology, Politics, Business, Sport, Entertainment. The following dataset can be found at the following url: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification

Technologies used: Python3, FastText, Jupyter-Notebook, pandas, matplotlib, seaborn

<p align="center"><img width="689" alt="Screenshot 2020-11-09 at 14 05 17" src="https://user-images.githubusercontent.com/11573356/98544747-b9a47880-2294-11eb-9cb6-6893722de5e4.png"></p>

## Core Functionalities:
```
  - Predict label
  - Get top k predictions
 ```
 
 ## Basic project installation steps:
```
  1. Clone repository
  
  2. Generate model & evaluation files:
     - preprocess pandas dataframe
     - import and create Evaluation object
     - create model using create_model() function
     - save model & evaluation files to a given output path
     
     Sample:
          from evaluation import Evaluation
          df = pd.read_csv('data/dataset.csv', sep="\t")
          ev = Evaluation(lang_code="en", method="FastText", version="1.1", clean_text=True, epoch=200, lr=1.0)
          ev.create_model(df, output_path="output")
    
     Evaluation files:
        - plot data distribution: train, test, original dataframe
        - plot sequence length
        - plot confusion matrix
        - plot wordclouds for each class in dataset
        - evaluation hyperparameters
        - classification report

  3. Predict label for new documents:
      - import and create Classifier object
      - predict label using predict_label() function

   Sample:
         from classifier import Classifier
         
         text = """
                As U.S. budget fight looms, Republicans flip their fiscal scriptWASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S.                   Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a “fiscal conservative” on Sunday and                 urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS’ “Face                 the Nation,” drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on                 Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as                 the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his                   Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense “discretionary”                       spending on programs that support education, scientific research, infrastructure, public health and environmental protection. “The (Trump)                         administration has already been willing to say: ‘We’re going to increase non-defense discretionary spending ... by about 7 percent,’” Meadows,                     chairman of the small but influential House Freedom Caucus, said on the program. “Now, Democrats are saying that’s not enough, we need to give the                 government a pay raise of 10 to 11 percent. For a fiscal conservative, I don’t see where the rationale is. ... Eventually you run out of other                     people’s money,” he said. Meadows was among Republicans who voted in late December for their party’s debt-financed tax overhaul, which is expected                 to balloon the federal budget deficit and add about $1.5 trillion over 10 years to the $20 trillion national debt. “It’s interesting to hear Mark                   talk about fiscal responsibility,” Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican tax bill would require                   the  United States to borrow $1.5 trillion, to be paid off by future generations, to finance tax cuts for corporations and the rich.
                """
                
         c = Classifier(lang_code="en", min_words=10, top_k=5, clean_text=False)
         pred = c.predict_label(text)
         print(pred)
         
         '''    
            {
                "label":"business",
                "confidence":0.8387,
                "predictions":[
                        {
                            "label":"business",
                            "confidence":0.8387
                        },
                        {
                            "label":"politics",
                            "confidence":0.1126
                        },
                        {
                            "label":"entertainment",
                            "confidence":0.0236
                        },
                        {
                            "label":"sport",
                            "confidence":0.0152
                        },
                        {
                            "label":"tech",
                            "confidence":0.01
                        }
                ],
                "message":"successful"
            }
         '''
       
```

## Classification report:
```
               precision    recall  f1-score   support

     business       0.97      0.95      0.96       103
entertainment       1.00      0.99      0.99        84
     politics       0.95      0.97      0.96        80
        sport       1.00      0.98      0.99        98
         tech       0.96      1.00      0.98        80

     accuracy                           0.98       445
    macro avg       0.98      0.98      0.98       445
 weighted avg       0.98      0.98      0.98       445
```
