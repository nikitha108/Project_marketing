![](images\stockimage1.jpg
)

# Customer Segmentation in Marketing Analysis

# Introduction

As a Data Analyst in the marketing domain, understanding customer behavior and tailoring marketing strategies accordingly is key to maximizing customer engagement and boosting sales. This project leverages KMeans clustering, a powerful machine learning technique, to segment customers into distinct groups based on their characteristics and behaviors. By clustering customers with similar attributes, businesses can identify valuable insights, enabling more personalized and effective marketing campaigns.

The dataset used in this project is sourced from Kaggle 
[Customer Segmentation Data
](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis) and includes key customer features such as age, gender, annual income, spending behavior, and purchase history. By applying KMeans clustering, we gain insights into customer segments that can inform better decision-making, targeted campaigns, and optimized customer retention strategies.


# Where can this analysis be used

1. Customer Segmentation : Identify distinct customer groups based on age, income, spending behavior, and preferences. Use these segments to understand customer diversity and create personalized experiences.

2. Targeted Marketing :Craft tailored marketing strategies for each segment, such as exclusive offers for high-value customers or promotions targeting specific age groups or preferred categories.

3. Product Recommendations : Leverage segmentation to provide personalized product recommendations, increasing the likelihood of cross-selling and upselling opportunities.

4. Churn Prediction and Prevention : Identify segments at risk of disengagement and implement targeted strategies, such as re-engagement campaigns or special incentives, to reduce churn.

5. Enhancing Customer Experience : Tailor communication, offers, and service levels based on segment-specific needs, leading to higher satisfaction and improved brand loyalty.

# Tools I Used

For my deep dive into the customer segmentation analysis, I harnessed the power of several key tools:

- **Python:** The backbone of my analysis, allowing me to analyze the data and find critical insights.I also used the following Python libraries:
    - **Pandas Library:** This was used to analyze the data. 
    - **Matplotlib Library:** I visualized the data.
    - **Seaborn Library:** Helped me create more advanced 
    visuals. 
    - **Scikit learn Library**Used to train the data to get clusters using Kmeans
- **Jupyter Notebooks:** The tool I used to run my Python scripts which let me easily include my notes and analysis.
- **Visual Studio Code:** My go-to for executing my Python scripts.
- **Git & GitHub:** Essential for version control and sharing my Python code and analysis, ensuring collaboration and project tracking.

# Data Preparation and Cleanup

This section outlines the steps taken to prepare the data for analysis, ensuring accuracy and usability.

## Import & Clean Up Data

I start by importing necessary libraries and loading the dataset, followed by initial data cleaning tasks to ensure data quality.

```python
#importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Loading Data

df = pd.read_csv("C:/Users/admin/Desktop/Portfolio projects/Project_marketing/dataset/customer_segmentation_data.csv", index_col= 'id')

```   
     
# Exploratory Data Analysis

To understand the structure and content of the dataset, I did :

1. Use the describe() to get the summary statistics 

```python

```

# The Analysis

Each Jupyter notebook for this project aimed at investigating specific aspects of the data job market. Here’s how I approached each question:

## 1. What are the most demanded skills for the top 3 most popular data roles?

To find the most demanded skills for the top 3 most popular data roles. I filtered out those positions by which ones were the most popular, and got the top 5 skills for these top 3 roles. This query highlights the most popular job titles and their top skills, showing which skills I should pay attention to depending on the role I'm targeting. 

View my notebook with detailed steps here: [skill_Demand](skills_demand.ipynb).

### Visualize Data

```python
fig , ax = plt.subplots(len(job_titles), 1)

sns.set_theme(style='ticks')

for i, job_title in enumerate(job_titles):
     df_plot = df_skills_percent[df_skills_percent['job_title_short']==job_title].head(5)
     #df_plot.plot(kind='barh',x='job_skills',y='skill_percent',ax=ax[i],title=job_title)
     sns.barplot(data=df_plot,x='skill_percent',y='job_skills',ax=ax[i],hue='skill_count',palette='dark:b_r')
     ax[i].set_title(job_title)
     ax[i].set_ylabel('')
     ax[i].set_xlabel('')
     ax[i].get_legend().remove()
     ax[i].set_xlim(0,70)

     for n,v in enumerate(df_plot['skill_percent']):
          ax[i].text(v + 1, n , f'{v:.0f}%',va='center')
     
     if i != len(job_titles)-1:
          ax[i].set_xticks([])



fig.suptitle(f'Likelihood of Skills Requested in {country} Job Postings', fontsize=13)
fig.tight_layout(h_pad=0.5)
plt.show()
```

### Results

![Likelihood of Skills Requested in France Job Postings](images\skills_demand.png)

*Bar graph visualizing the salary for the top 3 data roles and their top 5 skills associated with each.*

### Insights:

- Python is the most requested skill , highly demand across all three roles, but most promimently for Data Scientists(67%) and Data Engineers(57%).
- SQL is the most requested skill for Data Analysts and Data Engineers , with it in over half the job postings for both roles. 
- Data Engineers and Data Scientists require more specialized technicals skills (AWS, Azure, Spark, SAS) compared to Data Analysts who are expected to be more proficient in more general data management and analysis tools (Excel, tableau) 

## 2. How are in-demand skills trending for Data Analysts?

To find how skills are trending in 2023 for Data Analysts, I filtered data analyst positions and grouped the skills by the month of the job postings. This got me the top 5 skills of data analysts by month, showing how popular skills were throughout 2023.

View my notebook with detailed steps here: [skill_trend](skills_trends.ipynb).

### Visualize Data

```python

df_plot = df_percent.iloc[:,:5]
sns.lineplot(data=df_plot, dashes=False,palette='tab10')
sns.set_theme(style='ticks')
plt.title(f'Trending Top skills for {title}s in {country}')
plt.xlabel('2023')
plt.ylabel('Likelihood in Job Postings')
plt.legend().remove()
sns.despine()

from matplotlib.ticker import PercentFormatter
ax = plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))


for i in range(5):
    plt.text(11.2,df_plot.iloc[-1,i],df_plot.columns[i])

```

### Results

![Trending Top Skills for Data Analysts in France](images\skills_trend.png)  
*Line graph visualizing the trending top skills for data analysts in France in 2023.*

### Insights:
- SQL remains the most consistently demanded skill throughout the year. This highlights its importance as a core skill in France.
- Both Python and Tableau show relatively stable demand throughout the year with some fluctuations but remain essential skills for data analysts. 
- There are observable dips for skills during mid year(june-july), possibly indicating the seasonal hiring trendss  

## 3. How well do jobs and skills pay for Data Analysts?

To identify the highest-paying roles and skills, I only got jobs in France and looked at their median salary. But first I looked at the salary distributions of common data jobs like Data Scientist, Data Engineer, and Data Analyst, to get an idea of which jobs are paid the most.

View my notebook with detailed steps here: [salary_analysis](salary_analysis.ipynb).

#### Visualize Data 


```python
sns.boxplot(data=df_top6,x='salary_year_avg',y='job_title_short',order=job_order)
sns.set_theme(style='ticks')
plt.title('Salary Distribution in the United States')
plt.xlabel('Yearly Salary($USD)')
plt.ylabel('')
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'${int(y/1000)}K'))
plt.xlim(0,600000)

```

#### Results

The graph is not that appealing and doesnt look that informative. The salary is in USD in the data source while the currency in France is in Euro. This plot works accurately with only US job data.

![Salary Distributions of Data Jobs in France](images\salary_analysis.png)  
*Box plot visualizing the salary distributions for the top 6 data job titles.*

#### Insights

- There's a significant variation in salary ranges across different job titles.Senior Data Engineers earn the highest salaries, with tightly clustered pay for most professionals but notable outliers indicating exceptional earning potential

- The median salaries increase with the seniority and specialization of the roles. Senior roles (Senior Data Scientist, Senior Data Engineer) not only have higher median salaries but also larger differences in typical salaries, reflecting greater variance in compensation as responsibilities increase.

### Highest Paid & Most Demanded Skills for Data Analysts

Next, I narrowed my analysis and focused only on data analyst roles. I looked at the highest-paid skills and the most in-demand skills. I used two bar charts to showcase these.

#### Visualize Data

```python

fig, ax = plt.subplots(2, 1)
sns.set_theme(style='ticks')

#top paying skills
sns.barplot(data=df_top_pay, x='median',y= df_top_pay.index,hue='median',ax=ax[0],palette='dark:b_r')
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'${int(y/1000)}K'))

# top popular skills
sns.barplot(data=df_top_popular, x='median',y= df_top_popular.index,hue='median',ax=ax[1],palette='light:b')

ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'${int(y/1000)}K'))



fig.tight_layout()

```

#### Results
Here's the breakdown of the highest-paid & most in-demand skills for data analysts in France:

![The Highest Paid & Most In-Demand Skills for Data Analysts in France](images\top_pay_n_high_demand_skills.png)
*Two separate bar graphs visualizing the highest paid skills and most in-demand skills for data analysts in France*

#### Insights:

- The top graph shows specialized technical skills like `c`, `terraform`, and `Gitlab` are associated with higher salaries, some reaching up to $200K, suggesting that advanced technical proficiency can increase earning potential.

- The bottom graph highlights that foundational skills like `Excel`, `Python`, and `Powerbi` are the most in-demand, even though they may not offer the highest salaries. This demonstrates the importance of these core skills for employability in data analysis roles.

- There's a clear distinction between the skills that are highest paid and those that are most in-demand. Data analysts aiming to maximize their career potential should consider developing a diverse skill set that includes both high-paying specialized skills and widely demanded foundational skills.

## 4. What are the most optimal skills to learn for Data Analysts?

To identify the most optimal skills to learn ( the ones that are the highest paid and highest in demand) I calculated the percent of skill demand and the median salary of these skills. To easily identify which are the most optimal skills to learn. 

View my notebook with detailed steps here: [optimal_skills](optimal_skills.ipynb).

#### Visualize Data

```python
from adjustText import adjust_text
df_skills_high_demand.plot(kind='scatter', x='skill_percent',y= 'median_salary')


# Add labels to points and collect them in a list
texts = []
for i, txt in enumerate(df_skills_high_demand.index):
    texts.append(plt.text(df_skills_high_demand['skill_percent'].iloc[i], df_skills_high_demand['median_salary'].iloc[i], " " + txt))

# Adjust text to avoid overlap and add arrows
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))


# Get current axes, set limits, and format axes
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'${int(y/1000)}K')) 

from matplotlib.ticker import PercentFormatter
ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))

plt.tight_layout()
plt.show()

```

#### Results

![Most Optimal Skills for Data Analysts in France](images\optimal_skills.png)    
*A scatter plot visualizing the most optimal skills (high paying & high demand) for data analysts in France.*

#### Insights:

- `Python` appears to have the highest median salary of nearly $95K. This suggests a high value placed on Python , which is also a high demand skill(40%). This shows the importance of python for data anlaysts.

- Skills such as `Python`& `SQL` are towards the higher end of the salary spectrum while also being fairly common in job listings, indicating that proficiency in these tools can lead to good opportunities in data analytics.

### Visualizing Different Techonologies

Let's visualize the different technologies as well in the graph. We'll add color labels based on the technology (e.g., {Programming: Python})

#### Visualize Data

```python
sns.scatterplot(data=df_skills_tech_high_demand, x='skill_percent',y='median_salary',hue='technology')
sns.despine()
sns.set_theme(style='ticks')

from matplotlib.ticker import PercentFormatter
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'${int(y/1000)}K'))
ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))

# Adjust layout and display plot 
plt.tight_layout()
plt.show()

```

#### Results

![Most Optimal Skills for Data Analysts in France with Coloring by Technology](images\technology_colored.png)  
*A scatter plot visualizing the most optimal skills (high paying & high demand) for data analysts in France with color labels for technology.*

#### Insights:

- The scatter plot shows that most of the `programming` skills (colored blue) tend to cluster at higher salary levels compared to other categories, indicating that programming expertise might offer greater salary benefits within the data analytics field.

- The analyst_tools skills (colored orange), such as Excel and Power BI, are associated with some of the highest salaries among data analyst tools. This indicates a significant demand and valuation for data analysis and visualization expertise in the industry.


# What I Learned

Throughout this project, I deepened my understanding of the data analyst job market and enhanced my technical skills in Python, especially in data manipulation and visualization. Here are a few specific things I learned:

- **Advanced Python Usage**: Utilizing libraries such as Pandas for data manipulation, Seaborn and Matplotlib for data visualization, and other libraries helped me perform complex data analysis tasks more efficiently.
- **Data Cleaning Importance**: I learned that thorough data cleaning and preparation are crucial before any analysis can be conducted, ensuring the accuracy of insights derived from the data.
- **Strategic Skill Analysis**: The project emphasized the importance of aligning one's skills with market demand. Understanding the relationship between skill demand, salary, and job availability allows for more strategic career planning in the tech industry.


# Insights

This project provided several general insights into the data job market for analysts:

- **Skill Demand and Salary Correlation**: There is a clear correlation between the demand for specific skills and the salaries these skills command. Advanced and specialized skills like Python and AWS often lead to higher salaries.
- **Market Trends**: There are changing trends in skill demand, highlighting the dynamic nature of the data job market. Keeping up with these trends is essential for career growth in data analytics.
- **Economic Value of Skills**: Understanding which skills are both in-demand and well-compensated can guide data analysts in prioritizing learning to maximize their economic returns.


# Challenges I Faced

This project was not without its challenges, but it provided good learning opportunities:

- **Data Inconsistencies**: Handling missing or inconsistent data entries requires careful consideration and thorough data-cleaning techniques to ensure the integrity of the analysis.
- **Complex Data Visualization**: Designing effective visual representations of complex datasets was challenging but critical for conveying insights clearly and compellingly.
- **Balancing Breadth and Depth**: Deciding how deeply to dive into each analysis while maintaining a broad overview of the data landscape required constant balancing to ensure comprehensive coverage without getting lost in details.


# Conclusion

This exploration into the data analyst job market has been incredibly informative, highlighting the critical skills and trends that shape this evolving field. The insights I got enhance my understanding and provide actionable guidance for anyone looking to advance their career in data analytics. As the market continues to change, ongoing analysis will be essential to stay ahead in data analytics. This project is a good foundation for future explorations and underscores the importance of continuous learning and adaptation in the data field.As someone navigating the job market for good opportunities , this project was indeed very helpful!



















