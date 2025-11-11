import plotly.express as px
import plotly.graph_objects as go
import numpy as np, pandas as pd
def plot_leaderboard(df):
    d = df.copy()
    d["name"] = d["candidate_id"]
    fig = px.bar(d, x="name", y="final_score", hover_data=["jd_similarity","skill_coverage","exp_score","edu_score","recency_score"])
    fig.update_layout(yaxis_title="Final score", xaxis_title="Candidate", bargap=0.2)
    return fig
def plot_skill_coverage(found, missing):
    found_count = len(found) if isinstance(found, list) else 0
    missing_count = len(missing) if isinstance(missing, list) else 0
    fig = px.pie(values=[found_count, missing_count], names=["Found", "Missing"], hole=0.45)
    return fig
def plot_radar(row):
    cats = ["Similarity","Skills","Experience","Education","Recency"]
    vals = [row["jd_similarity"], row["skill_coverage"], row["exp_score"], row["edu_score"], row["recency_score"]]
    fig = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill="toself"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False)
    return fig