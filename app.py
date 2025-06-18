from flask import Flask, request, jsonify
import pandas as pd
import joblib, pickle

app = Flask(__name__)

# 🧠 Carga modelos y transformadores
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("modelo_randomforest.joblib")
mlb_req_skills = joblib.load("mlb_req_skills.pkl")
ohe = joblib.load("ohe.pkl")
constant_filter = joblib.load("constant_filter.pkl")
selector = joblib.load("selector.pkl")

# Carga de datos base
df = pd.read_csv("df_XGB.csv")
df["skills"] = df["skills"].apply(eval)
df["required_skills"] = df["required_skills"].apply(eval)

def preparar_X(candidatos, skills, categoria, complejidad, rol):
    candidatos["proyecto_skills"] = [skills] * len(candidatos)
    candidatos["skill_match"] = candidatos.apply(
        lambda row: len(set(row["skills"]) & set(skills)) / len(skills)
        if skills else 0, axis=1
    )
    candidatos["xgb_senior_match"] = ((candidatos["experience_years"] >= 5) &
                                      (candidatos["skill_match"] >= 0.7)).astype(int)
    candidatos["complexity"] = complejidad
    candidatos["category"] = categoria

    req_skills = pd.DataFrame(
        mlb_req_skills.transform(candidatos["proyecto_skills"]),
        columns=[f"req_skill_{s}" for s in mlb_req_skills.classes_]
    )
    cat_df = pd.DataFrame(
        ohe.transform(candidatos[["role", "assigned_role", "category"]]),
        columns=ohe.get_feature_names_out()
    )
    X = pd.concat([
        candidatos[["experience_years","complexity","skill_match","xgb_senior_match"]].reset_index(drop=True),
        req_skills.reset_index(drop=True),
        cat_df.reset_index(drop=True)
    ], axis=1)
    X = X.reindex(columns=constant_filter.feature_names_in_, fill_value=0)
    X = constant_filter.transform(X)
    return selector.transform(X)

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "API de recomendaciones activa", "endpoint": "/recomendar"})

@app.route("/recomendar", methods=["POST"])
def recomendar():
    data = request.get_json()
    categoria = data.get("categoria")
    complejidad = int(data.get("complejidad"))
    rol = data.get("rol")
    skills = data.get("skills", [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]

    if not all([categoria, rol, skills]) or complejidad is None:
        return jsonify({"status":"error","message":"Faltan datos: categoría, rol, skills o complejidad"}),400

    candidatos = df[df["assigned_role"] == rol].copy()
    X = preparar_X(candidatos, skills, categoria, complejidad, rol)

    # 1️⃣ Intento con XGBoost
    preds_xgb = xgb_model.predict(X)
    candidatos["pred_xgb"] = preds_xgb

    if candidatos["pred_xgb"].sum() > 0:
        rec = candidatos[candidatos["pred_xgb"] == 1]
        rec = rec.drop_duplicates("employee_id").sort_values(
            by=["availability","skill_match","experience_years"], ascending=False
        )
        return jsonify({"status":"ok","modelo":"xgboost","recomendaciones":rec[["employee_id","assigned_role","experience_years","availability","skill_match"]].to_dict(orient="records")})

    # 2️⃣ Si XGB no encuentra, usar RandomForest
    preds_rf = rf_model.predict(X)
    candidatos["pred_rf"] = preds_rf
    if candidatos["pred_rf"].sum() == 0:
        return jsonify({"status":"ok","mensaje":"No hay candidatos viables en ninguno de los modelos"})

    rec = candidatos[candidatos["pred_rf"] == 1]
    rec = rec.drop_duplicates("employee_id").sort_values(
        by=["availability","skill_match","experience_years"], ascending=False
    )
    return jsonify({"status":"ok","modelo":"randomforest","recomendaciones":rec[["employee_id","assigned_role","experience_years","availability","skill_match"]].to_dict(orient="records")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)