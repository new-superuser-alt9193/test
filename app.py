
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar modelo y transformadores
model = joblib.load("xgb_model.pkl")
mlb_req_skills = joblib.load("mlb_req_skills.pkl")
ohe = joblib.load("ohe.pkl")
constant_filter = joblib.load("constant_filter.pkl")
selector = joblib.load("selector.pkl")

# Cargar dataframe original
df = pd.read_csv("df_XGB.csv")
df["skills"] = df["skills"].apply(eval)
df["required_skills"] = df["required_skills"].apply(eval)

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "API de recomendaciones activa", "endpoint": "/recomendar"})

@app.route("/recomendar", methods=["POST"])
def recomendar():
    try:
        data = request.get_json()
        categoria = data.get("categoria")
        complejidad = int(data.get("complejidad"))
        rol = data.get("rol")
        skills = data.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
 # Validar entrada
        if not categoria or not rol or not skills or complejidad is None:
            return jsonify({
                "status": "error",
                "message": "Faltan uno o más datos requeridos: categoría, rol, skills o complejidad"
            })
        candidatos = df[df["assigned_role"] == rol].copy()
        candidatos["proyecto_skills"] = [skills] * len(candidatos)

        candidatos["skill_match"] = candidatos.apply(
            lambda row: len(set(row["skills"]) & set(row["proyecto_skills"])) / len(row["proyecto_skills"])
            if len(row["proyecto_skills"]) > 0 else 0, axis=1
        )
        candidatos["xgb_senior_match"] = ((candidatos["experience_years"] >= 5) &
                                          (candidatos["skill_match"] >= 0.7)).astype(int)
        candidatos["complexity"] = complejidad
        candidatos["category"] = categoria

        # Generar variables de entrada para el modelo
        req_skills_proj_df = pd.DataFrame(
            mlb_req_skills.transform(candidatos["proyecto_skills"]),
            columns=[f"req_skill_{s}" for s in mlb_req_skills.classes_]
        )

        cat_proj_df = pd.DataFrame(
            ohe.transform(candidatos[["role", "assigned_role", "category"]]),
            columns=ohe.get_feature_names_out()
        )

        X = pd.concat([
            candidatos[["experience_years", "complexity", "skill_match", "xgb_senior_match"]].reset_index(drop=True),
            req_skills_proj_df.reset_index(drop=True),
            cat_proj_df.reset_index(drop=True)
        ], axis=1)

        X = X.reindex(columns=constant_filter.feature_names_in_, fill_value=0)
        X = constant_filter.transform(X)
        X = selector.transform(X)

        candidatos["prediccion_alto"] = model.predict(X)
        recomendados = candidatos[candidatos["prediccion_alto"] == 1]

        if len(recomendados) == 0:
            # Alternativa: mostrar los 3 mejores si no hay candidatos "Alto"
            alternativos = candidatos.sort_values(
                by=["skill_match", "experience_years"], ascending=False
            ).drop_duplicates(subset="employee_id").head(3)[
                ["employee_id", "assigned_role", "experience_years", "availability", "skill_match"]
            ]
            return jsonify({
                "status": "ok",
                "mensaje": "No se encontraron candidatos con clasificación 'Alto'",
                "alternativas": alternativos.to_dict(orient="records")
            })

        # Ordenar candidatos y eliminar duplicados
        top = recomendados.drop_duplicates(subset="employee_id").sort_values(
            by=["availability", "skill_match", "experience_years"],
            ascending=[False, False, False]
        )[
            ["employee_id", "assigned_role", "experience_years", "availability", "skill_match"]
        ]

        return jsonify({"status": "ok", "recomendaciones": top.to_dict(orient="records")})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
