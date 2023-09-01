#!/usr/bin/env python
# coding: utf-8

# # **Pokemon dataset analysis**

# ## **What are Pokémon?**
# 
# Pokémon are like magical creatures that live in a world alongside humans. They come in all shapes and sizes, from tiny ones that fit in your hand to big ones that are as tall as a building. Each Pokémon is unique and has its own special abilities and characteristics. Some can shoot fire, others can fly, and some are really good at swimming. People in this world become friends with Pokémon and train them to be strong so they can have exciting adventures together. It's like having your own team of amazing animal friends that you can play and have fun with.**

# ## **Pokédex Data Summary:**
# 
# The Pokémon Pokedex is like a big book that contains information about all kinds of creatures called Pokémon. It's like a big encyclopedia full of facts about these special animals. Each Pokémon has its own page with details about how it looks, its abilities, strengths, weaknesses, and more. The Pokedex helps trainers (people who catch and train Pokémon) understand and learn about different Pokémon so they can become experts in caring for and battling with them.**

# ## **Objective:**
# 
# Analyze Pokedex data containing information about Pokemon across 8 generations to learn about various Pokemon attributes like strengths, weakness, physical features, catch rate etc. Based on the findings, we will build a model to predict catch rate for a customized pokemon. The above analysis can help achieve the following:
# * **Understand world of Pokemon**
# * **Use this as a guide for Pokemon battles in games**
#     * which pokemon to use against the opponent Pokemon
#     * How to build invincible team of pokemon
#     

# ## **Data Dictionary:**
# 
# The data contains various information for Pokemon.** The detailed data dictionary is given below:
# 
# 
# Pokedex Data:
# 
#    * **pokedex_number**: The entry number of the Pokemon in the National Pokedex
#    * **name**: The English name of the Pokemon
#    * **german_name**: The German name of the Pokemon
#    * **japanese_name**: The Original Japanese name of the Pokemon
#    * **generation**: The numbered generation which the Pokemon was first introduced,
#    * **status**: Denotes if the Pokemon is normal, sub-legendary, legendary or mythical
#    * **type_number**: Number of types that the Pokemon has
#    * **type_1**: The Primary Type of the Pokemon
#    * **type_2**: The Secondary Type of the Pokemon if it has it
#    * **height_m**: Height of the Pokemon in meters
#    * **weight_kg**: The Weight of the Pokemon in kilograms
#    * **abilities_number**: The number of abilities of the Pokemon
#    * **ability_?**: Name of the Pokemon abilities
#    * **ability_hidden**: Name of the hidden ability of the Pokemon if it has one
# 
# Base stats:
# 
#    * **total_points**: Total number of Base Points
#    * **hp**: The Base HP of the Pokemon
#    * **attack**: The Base Attack of the Pokemon
#    * **defense**: The Base Defense of the Pokemon
#    * **sp_attack**: The Base Special Attack of the Pokemon
#    * **sp_defense**: The Base Special Defense of the Pokemon
#    * **speed**: The Base Speed of the Pokemon
# 
# Training:
# 
#    * **catch_rate**: Catch Rate of the Pokemon (Lower the catch rate, harder it is to catch the pokemon)
#    * **base_friendship**: The Base Friendship of the Pokemon
#    * **base_experience**: The Base experience of a wild Pokemon when caught
#    * **growth_rate**: The Growth Rate of the Pokemon
# 
# Breeding:
# 
#    * **egg_type_number**: Number of groups where a Pokemon can hatch
#    * **egg_type_?**: Names of the egg groups where a Pokemon can hatch
#    * **percentage_male**: The percentage of the species that are male. Blank if the Pokemon is genderless.
#    * **egg_cycles**: The number of cycles (255-257 steps) required to hatch an egg of the Pokemon
# 
# Type defenses:
# 
#    * **against_?**: Eighteen features that denote the amount of damage taken against an attack of a particular type
# 

# ## **Additional data used**
# 
# Generation 8 images were imported after webscraping from public pokemon sprite gallery (https://pokemondb.net/sprites) for pokemon listed on the webpage. This data would help us visualize the appearance of different pokemon**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import networkx as nx
import os
import warnings
#import requests
#from bs4 import BeautifulSoup
import random
warnings.filterwarnings("ignore")

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report,recall_score,precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For tuning the model
from sklearn.model_selection import GridSearchCV

# Algorithms to use
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

# Metrics to evaluate the model
from sklearn import metrics

# For tuning the model
from sklearn.model_selection import GridSearchCV

# To build models for prediction
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

# To encode categorical variables
from sklearn.preprocessing import LabelEncoder

# For tuning the model
from sklearn.model_selection import GridSearchCV

# To check model performance
from sklearn.metrics import make_scorer,mean_squared_error, r2_score, mean_absolute_error

from IPython.display import HTML


# In[2]:


# URL of the raw CSV file on GitHub
csv_url = "https://raw.githubusercontent.com/manansatsangi/Pokemon/master/pokedex_(Update_05.20).csv"

# Read the CSV file into a pandas DataFrame
df_pokemon = pd.read_csv(csv_url)


# In[3]:


against=[ 'against_normal',
 'against_fire',
 'against_water',
 'against_electric',
 'against_grass',
 'against_ice',
 'against_fight',
 'against_poison',
 'against_ground',
 'against_flying',
 'against_psychic',
 'against_bug',
 'against_rock',
 'against_ghost',
 'against_dragon',
 'against_dark',
 'against_steel',
 'against_fairy']

base_stats=['total_points',
 'hp',
 'attack',
 'defense',
 'sp_attack',
 'sp_defense',
 'speed']

egg=['egg_type_number',
 'egg_type_1',
 'egg_type_2',
 'percentage_male',
 'egg_cycles']

phys_stats=['height_m', 'weight_kg']

training=[ 'catch_rate', 'base_friendship', 'base_experience','growth_rate']

details=['generation',
 'status',
 'species',
 'type_number',
 'type_1',
 'type_2',
 'abilities_number',
 'ability_1',
 'ability_2',
 'ability_hidden']


# ## Missing values treatment

# The pokedex dataset has missing values which were treated the following way:
# Categorical data replaced with 'Unknown' and 'None' where applicable
# Numerical data replaced with zero

# In[4]:


df_poke=df_pokemon.copy()
df_poke['egg_type_2'].fillna('None', inplace = True)
df_poke['ability_2'].fillna('None', inplace = True)
df_poke['type_2'].fillna('None', inplace = True)
df_poke['percentage_male'].fillna(0, inplace = True)
df_poke['base_friendship'].fillna(0, inplace = True)
df_poke['base_experience'].fillna(0, inplace = True)
df_poke['catch_rate'].fillna(0, inplace = True)
df_poke['japanese_name'].fillna('Unknown', inplace = True)
df_poke['german_name'].fillna('Unknown', inplace = True)
df_poke['egg_type_1'].fillna('Unknown', inplace = True)
df_poke['ability_1'].fillna('Unknown', inplace = True)
df_poke['growth_rate'].fillna('Unknown', inplace = True)
df_poke['weight_kg'].fillna(0, inplace = True)
df_poke['egg_cycles'].fillna(0, inplace = True)


# In[5]:


# Create empty lists to store weak against and effective against types
weak_against = []
effective_against = []

attrib = ['against_normal',
             'against_fire',
             'against_water',
             'against_electric',
             'against_grass',
             'against_ice',
             'against_fight',
             'against_poison',
             'against_ground',
             'against_flying',
             'against_psychic',
             'against_bug',
             'against_rock',
             'against_ghost',
             'against_dragon',
             'against_dark',
             'against_steel',
             'against_fairy']

# Iterate through each row in the DataFrame
for index, row in df_poke.iterrows():
    # Reset lists for each Pokemon
    weak_against = []
    effective_against = []
    
    # Iterate through the attributes
    for att in attrib:
        # Get the value of the current attribute
        value = row[att]
        
        # Check if the Pokemon is weak or effective against this type
        if value < 1:
            effective_against.append(att[8:].replace('_', ' ').capitalize())  # Removing 'against_' and formatting
        elif value > 1:
            weak_against.append(att[8:].replace('_', ' ').capitalize())  # Removing 'against_' and formatting
    
    # Update the DataFrame with the lists of weak and effective against types
    df_poke.at[index, 'Effective_against'] = ', '.join(effective_against)
    df_poke.at[index, 'Weak_against'] = ', '.join(weak_against)


# In[6]:


def plotx(z):
    total = len(df_poke[z])
    plt.figure(figsize=(10,5))
    ax=sns.countplot(x=df_poke[z],
                 order = df_poke[z].value_counts().index,
                 palette="viridis")
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total) # Percentage of each class
        x = p.get_x() + p.get_width() / 2 - 0.05                    # Width of the plot
        y = p.get_y() + p.get_height()                              # Height of the plot
        ax.annotate(percentage, (x, y), size = 12)                  # Annotate the percentage 
    plt.xticks(rotation=90)
    plt.title('Pokémon distribution by ' + z)
    #plt.xlabel('Generation')
    #plt.ylabel('Base Stat')
    #plt.legend(title='Base Stat', loc='upper left')
    #plt.xticks(grouped_data.index)  # Set x-axis labels to generation numbers   


# In[7]:


def ptype(z,x):
    plt.figure(figsize=(13,8))
    sns.boxplot(y=df_poke[z],
                x=df_poke[x],
               palette="viridis",
               showmeans = True)
    plt.grid(True)
    plt.title("Pokemon "+z+" by " +x,color='b')
    plt.xticks(rotation=90)
    plt.show()


# In[8]:


def base_stats(x):
    grouped_data = df_poke.groupby(x)[[ 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].mean()
    plt.figure(figsize=(12, 8))
    sns.set_palette("viridis")
    sns.lineplot(data=grouped_data, markers=True, marker='o', markersize=8)  # Use circles with larger size

    plt.title('Pokémon Base Stats by ' + x)
    plt.xlabel(x)
    plt.ylabel('Average Base Stat')
    plt.legend(title='Base Stat', bbox_to_anchor=(1, 1))
    plt.xticks(grouped_data.index)
    plt.grid(True)

    plt.show()


# In[9]:


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a sorted stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    chart_title: title for the chart
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    #print(tab1)
    #print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    
    # Sort the DataFrame by the predictor in ascending order
    sorted_indices = tab.index.sort_values()
    tab = tab.loc[sorted_indices]
    
    # Set the color palette to "viridis"
    sns.set_palette("viridis")
    
    # Plot the stacked bar chart
    ax = tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    
    # Add chart title
    ax.set_title("Pokemon "+ target +" by "+ predictor)
    
    # Customize legends
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# In[10]:


def heat(x):
    # Pivot the data to create a heatmap
    heatmap_data = df_poke[df_poke['type_number']==1].pivot_table(index=x, values=['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed'])

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data=heatmap_data, annot=True, fmt=".1f", cmap="viridis", linewidths=0.5,cbar_kws={'label': 'Average Base Stat'})

    plt.title('Pokémon Average Base Stats by '+x)
    plt.xlabel('Base Stat')
    plt.ylabel(x)

    plt.show()


# In[11]:


def weak(x):
    # Pivot the data to create a heatmap
    heatmap_data = df_poke[df_poke['type_number']==1].pivot_table(index=x, values=against)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data=heatmap_data, annot=True, fmt=".1f", cmap="viridis", linewidths=0.5,cbar_kws={'label': 'Average against stat'})

    plt.title('Pokémon weakness by '+x)
    plt.xlabel('Against type')
    plt.ylabel(x)

    plt.show()


# In[12]:


def phys(x):
    from sklearn.preprocessing import StandardScaler

    # Assuming you have already defined 'df_poke', 'x', and 'phys_stats'

    # Create a pivot table
    heatmap_data = df_poke.pivot_table(index=x, values=phys_stats)

    # Normalize the values using StandardScaler
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(heatmap_data)

    # Convert the normalized array back to a DataFrame
    normalized_heatmap_data = pd.DataFrame(normalized_values, columns=heatmap_data.columns, index=heatmap_data.index)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data=normalized_heatmap_data, annot=True, fmt=".1f", cmap="viridis", linewidths=0.5,cbar_kws={'label': 'Average Base Stat (Normalized)'})

    plt.title('Pokémon Average Physical Stats (Normalized) by '+x)
    plt.xlabel('Base Stat')
    plt.ylabel(x)

    plt.show()


# In[13]:


def catch(x):
    grouped_data = df_poke.groupby(x)[['catch_rate','base_friendship','base_experience']].mean()

    plt.figure(figsize=(12, 8))

    sns.set_palette("viridis")  # Set color palette
    sns.lineplot(data=grouped_data, markers=True)

    plt.title('Pokémon Stats by '+x)
    plt.xlabel(x)
    plt.ylabel('Stat')
    #plt.legend(title='Base Stat', loc='upper left')
    plt.xticks(grouped_data.index)  # Set x-axis labels to generation numbers

    plt.show()


# ## Lets try to understand the world of pokemon

# In[15]:


# Set the color palette to "viridis"
sns.set_palette("viridis")


# ### Pokemon Stats analysis

# * The **Hit points, attack, special attack, defense, special defense and speed** are the pokemon stats that usually impact pokemon battles
# * There appears to be a positive correlation between total points and the above stats. **Higher the base stats, higher the total points** for pokemon.

# In[16]:


plt.figure(figsize=(12, 10))
correlation_matrix = df_poke[['total_points', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="viridis", center=0)
plt.title('Correlation Heatmap')
plt.show()


# ### Pokemon Primary type analysis

# * **Water, Normal and Grass** are the most common pokemon types in the world of pokemon
# * **Flying and Fairy** are the rarest

# In[17]:


plotx('type_1')


# ### Stat analysis:
# * **Attack (Higherst/Lowest)**: Fighting / Bug
# * **Special attack (Higherst/Lowest)**: Psychic / Bug
# * **Defense (Higherst/Lowest)**: Steel / Bug
# * **Special Defense (Higherst/Lowest)**: Ghost / Bug
# * **Speed (Higherst/Lowest)**: Flying / Steel
# * **Base HP (Higherst/Lowest)**: Normal / Ghost

# In[18]:


heat('type_1')


# ### Weakness analysis
# * The **against_type** attribute gives us an idea about a pokemon's **defenses** against an attack from that pokemon type.
# * **Higher** the value of **against_type**, the **weaker** the pokemon will be against that type
# * For example, typically, a pokemon with **Dark type** as primary type is weak against **Bug, Fairy and Fighting** pokemon.
# * On the other hand, **Dark type** pokemon is effective against **Ghost and other Dark** pokemon.
# * Dark type pokemon is not at all affected by **Psychic type** moves

# In[19]:


weak('type_1')


# ### Catch rate analysis
# * The **Dragon, Flying and Steel** pokemon are the most difficult to catch
# * The **Normal, Poison and Bug** pokemon are relatively easier to catch

# In[20]:


ptype("catch_rate","type_1")


# ### Physical stats analysis
# * The **Poison and Dragon** pokemon are the tallest
# * The **Bug and Fairy** pokemon are the smallest
# * The **Steel and Ground** pokemon are relatively heavier.
# * The **Bug and Electric** pokemon are relatively lighter.

# In[21]:


phys('type_1')


# ### Growth rate analysis
# * The **Fairy and Ghost** type pokemon grow relatively faster
# * The **Dragon and Steel** type pokemon grow relatively slower

# In[22]:


stacked_barplot(df_poke, "type_1", "growth_rate")


# ### Hatch rate analysis
# * On average, the **Bug, Normal and Ghost** type pokemon hatch faster from their eggs
# * The **Dragon, Psychic and Flying** type pokemon take longer to hatch from their eggs

# In[23]:


ptype("egg_cycles","type_1")

