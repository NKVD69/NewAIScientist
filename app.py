import streamlit as st
import asyncio
import concurrent.futures
import threading
import pandas as pd
import json
import os
import sys
import hashlib
import logging
from datetime import datetime
import plotly.express as px
from dataclasses import asdict
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Defensive definitions for potential JS-style JSON/boolean errors
true = True
false = False
null = None

# Configure logging
logger = logging.getLogger(__name__)

from co_scientist import CoScientist, ResearchGoal, Hypothesis, get_llm_usage_stats

# ---------------------------------------------------------------------------
# ASYNC HELPER — run coroutines safely in a dedicated thread with its own loop
# ---------------------------------------------------------------------------

def run_async(coro):
    """
    Run an async coroutine in a dedicated thread with its own event loop.
    Avoids ProactorEventLoop conflicts on Windows and Streamlit's internal loop.
    """
    ctx = get_script_run_ctx()

    def _run():
        if ctx:
            add_script_run_ctx(threading.current_thread(), ctx)
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        return future.result()  # Blocks until coroutine completes


class StreamlitRedirector:
    """Capture stdout/stderr and render into a Streamlit code element."""
    def __init__(self, text_element):
        self.text_element = text_element
        self.buffer = ""
        self.last_update_time = 0.0
        self.ctx = get_script_run_ctx()

    def write(self, msg):
        if self.ctx and not get_script_run_ctx():
            add_script_run_ctx(threading.current_thread(), self.ctx)
            
        self.buffer += msg
        if len(self.buffer) > 10000:
            self.buffer = self.buffer[-10000:]
        
        # Debounce UI updates to max 10 FPS to avoid freezing Streamlit
        import time
        now = time.time()
        if now - self.last_update_time > 0.1:
            try:
                self.text_element.code(self.buffer, language="log")
                self.last_update_time = now
            except Exception:
                pass

    def flush(self):
        if self.ctx and not get_script_run_ctx():
            add_script_run_ctx(threading.current_thread(), self.ctx)
        try:
            self.text_element.code(self.buffer, language="log")
        except Exception:
            pass

    def isatty(self):
        """Return False to indicate this is not a real terminal."""
        return False


CONFIG_FILE = "app_config.json"
SESSIONS_DIR = "sessions"


def load_config():
    """Load configuration from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Erreur lors du chargement de la config: {e}")
    return {}


def save_config(params):
    """Save configuration to file"""
    try:
        config = load_config()
        config.update(params)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la config: {e}")


def save_session(cs: CoScientist, goal_title: str) -> str:
    """Persist current co-scientist state to a timestamped JSON file. Returns path."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() else "_" for c in goal_title)[:40]
    path = os.path.join(SESSIONS_DIR, f"session_{ts}_{safe_title}.json")
    hypotheses = list(cs.context_memory.hypotheses.values())
    data = {
        "saved_at": datetime.now().isoformat(),
        "goal": asdict(cs.context_memory.research_goal),
        "literature_context": cs.context_memory.literature_context,
        "hypotheses": [asdict(h) for h in hypotheses],
        "tournament_history": [asdict(m) for m in cs.context_memory.tournament_history],
        "meta_reviews": cs.context_memory.meta_reviews,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    logger.info("Session saved: %s", path)
    return path


def list_sessions() -> list:
    """Return list of saved session files, newest first."""
    if not os.path.isdir(SESSIONS_DIR):
        return []
    files = [
        os.path.join(SESSIONS_DIR, f)
        for f in os.listdir(SESSIONS_DIR)
        if f.endswith(".json") and f.startswith("session_")
    ]
    return sorted(files, reverse=True)


# Load initial config
config = load_config()

# Configuration de la page
st.set_page_config(
    page_title="AI Co-Scientist Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés pour un look plus moderne
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .hypothesis-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #3b82f6;
    }
    .paper-card {
        background-color: #2d3748;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #4a5568;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # LLM Settings
    st.subheader("LLM Settings")
    use_local_llm = st.checkbox("Utiliser un LLM local",
                                value=config.get("use_local_llm", True),
                                key="persist_use_local_llm",
                                on_change=lambda: save_config({"use_local_llm": st.session_state.persist_use_local_llm}))

    if use_local_llm:
        llm_base_url = st.text_input("URL du LLM",
                                   value=config.get("llm_base_url", "http://127.0.0.1:1234/v1"),
                                   key="persist_llm_base_url",
                                   on_change=lambda: save_config({"llm_base_url": st.session_state.persist_llm_base_url}))
        llm_model_name = st.text_input("Nom du modèle",
                                     value=config.get("llm_model_name", "openai/gpt-oss-20b"),
                                     key="persist_llm_model_name",
                                     on_change=lambda: save_config({"llm_model_name": st.session_state.persist_llm_model_name}))

        # Mise à jour des variables d'environnement pour le backend
        os.environ["OPENAI_API_BASE"] = llm_base_url
        os.environ["OPENAI_API_KEY"] = "lm-studio"  # Dummy key for local
        os.environ["OPENAI_MODEL_NAME"] = llm_model_name

        # BOUTON DE TEST DE CONNEXION
        if st.button("📡 Tester la connexion"):
            try:
                import openai
                client = openai.OpenAI(base_url=llm_base_url, api_key="lm-studio")
                with st.spinner("Ping du serveur..."):
                    models = client.models.list()
                    st.success(f"Connexion réussie ! Serveur actif.")
            except Exception as e:
                st.error(f"Échec de connexion : {e}")
    else:
        st.warning("Mode Simulation activé")

    st.divider()
    st.subheader("Paramètres de Recherche")
    num_hypotheses = st.slider("Nombre d'hypothèses", 3, 20,
                                value=config.get("num_hypotheses", 5),
                                key="persist_num_hyp",
                                on_change=lambda: save_config({"num_hypotheses": st.session_state.persist_num_hyp}))
    num_iterations = st.slider("Nombre d'itérations", 1, 10,
                                value=config.get("num_iterations", 3),
                                key="persist_num_iter",
                                on_change=lambda: save_config({"num_iterations": st.session_state.persist_num_iter}))

    st.subheader("Paramètres Sources")
    max_papers = st.slider("Max Papiers/Source", 3, 100,
                            value=config.get("max_papers", 5),
                            key="persist_max_papers",
                            on_change=lambda: save_config({"max_papers": st.session_state.persist_max_papers}))

    st.divider()

    # RAG System Settings
    enable_rag = st.checkbox("Activer RAG (téléchargement & analyse PDF)",
                              value=config.get("enable_rag", True),
                              key="persist_enable_rag",
                              on_change=lambda: save_config({"enable_rag": st.session_state.persist_enable_rag}),
                              help="Télécharge les PDFs et effectue une recherche sémantique avancée")

    # Display RAG stats if available
    if 'results' in st.session_state and st.session_state.results is not None:
        try:
            rag_stats = st.session_state.results.literature_agent.get_rag_stats()
            if rag_stats['status'] == 'ready':
                st.success(f"🧠 RAG actif: {rag_stats.get('total_chunks', 0)} chunks indexés")
            elif rag_stats['status'] == 'disabled':
                st.info("ℹ️ RAG désactivé")
        except (AttributeError, KeyError):
            pass

    st.divider()

    # LLM Usage Stats
    usage = get_llm_usage_stats()
    if usage.get("total_calls", 0) > 0:
        st.subheader("📊 Stats LLM")
        st.metric("Appels LLM", usage["total_calls"])
        st.metric("Tokens Consommés", f"{usage['total_tokens']:,}")
        if usage.get("last_model"):
            st.caption(f"Modèle: `{usage['last_model']}`")

    st.divider()

    # Authentification PubMed / NCBI
    st.subheader("🔐 Auth PubMed (NCBI)")
    st.markdown("Requis pour augmenter la limite de requêtes (10/s) avec PubMed/PMC.")
    entrez_email = st.text_input("Email Entrez",
                                value=config.get("entrez_email", "ai-scientist@example.com"),
                                key="persist_entrez_email",
                                on_change=lambda: save_config({"entrez_email": st.session_state.persist_entrez_email}))

    ncbi_api_key = st.text_input("Clé API NCBI (optionnel)",
                                type="password",
                                value=config.get("ncbi_api_key", ""),
                                key="persist_ncbi_api_key",
                                on_change=lambda: save_config({"ncbi_api_key": st.session_state.persist_ncbi_api_key}))

    os.environ["ENTREZ_EMAIL"] = entrez_email
    if ncbi_api_key:
        os.environ["NCBI_API_KEY"] = ncbi_api_key

    try:
        from Bio import Entrez
        Entrez.email = entrez_email
        if ncbi_api_key:
            Entrez.api_key = ncbi_api_key
    except ImportError:
        pass

    st.divider()

    # Session persistence — load previous session
    st.subheader("💾 Sessions Sauvegardées")
    saved_sessions = list_sessions()
    if saved_sessions:
        session_labels = [os.path.basename(s) for s in saved_sessions]
        selected_session = st.selectbox("Charger une session", ["(nouvelle session)"] + session_labels)
        if st.button("📂 Charger") and selected_session != "(nouvelle session)":
            session_path = saved_sessions[session_labels.index(selected_session)]
            try:
                with open(session_path, "r", encoding="utf-8") as f:
                    _data = json.load(f)
                st.success(f"Session chargée: {selected_session}")
                st.session_state["loaded_session_data"] = _data
                st.rerun()
            except Exception as e:
                st.error(f"Erreur de chargement: {e}")
    else:
        st.caption("Aucune session sauvegardée.")


# --- CONFIGURATION & SESSION STATE ---
if "persist_goal_title" not in st.session_state:
    st.session_state.persist_goal_title = config.get("goal_title", "Drug Repurposing for AML")
if "persist_goal_domain" not in st.session_state:
    st.session_state.persist_goal_domain = config.get("goal_domain", "Biomedicine/Oncology")
if "persist_goal_desc" not in st.session_state:
    st.session_state.persist_goal_desc = config.get("goal_desc", "Identify FDA-approved drugs that could be repurposed for acute myeloid leukemia (AML) treatment.")
if "persist_sources" not in st.session_state:
    st.session_state.persist_sources = config.get("source_type", ["arxiv", "pubmed"])
if "persist_constraints" not in st.session_state:
    st.session_state.persist_constraints = config.get("constraints", "Only FDA-approved drugs\nMust have mechanism documentation")

# --- MAIN CONTENT ---

st.title("🧬 AI Co-Scientist Workbench")
st.markdown("### Assistant de découverte scientifique multi-agents")

# Initialisation de l'état de session
if 'co_scientist' not in st.session_state:
    st.session_state.co_scientist = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# --- SECTION 1: DEFINITION DE L'OBJECTIF ---
with st.expander("🎯 Définir l'Objectif de Recherche", expanded=not st.session_state.is_running):
    if "suggested_domain" in st.session_state:
        st.session_state.persist_goal_domain = st.session_state.suggested_domain
        del st.session_state.suggested_domain
    if "suggested_sources" in st.session_state:
        st.session_state.persist_sources = st.session_state.suggested_sources
        del st.session_state.suggested_sources

    col1, col2 = st.columns([1, 1])

    with col1:
        goal_title = st.text_input("Titre de la Recherche",
                                 key="persist_goal_title",
                                 on_change=lambda: save_config({"goal_title": st.session_state.persist_goal_title}))

        col_dom, col_btn = st.columns([3, 1])
        with col_dom:
            goal_domain = st.text_input("Domaine Scientifique",
                                      key="persist_goal_domain")
        with col_btn:
            st.write("")  # Spacer
            if st.button("🪄 Auto", help="Détecter le domaine et les sources via IA"):
                with st.spinner("Analyse..."):
                    try:
                        temp_cs = CoScientist(use_local_llm=use_local_llm, enable_rag=enable_rag)
                        analysis = run_async(temp_cs.analyze_research_description(st.session_state.persist_goal_desc))
                        if analysis:
                            st.session_state.suggested_domain = ", ".join(analysis.get("domains", []))
                            st.session_state.suggested_sources = [
                                s for s in analysis.get("databases", [])
                                if s in ["arxiv", "pubmed", "biorxiv", "ieee_xplore", "scopus", "google_scholar", "semantic_scholar"]
                            ]
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erreur d'analyse: {e}")

    with col2:
        goal_desc = st.text_area("Description Détaillée",
                               height=100,
                               key="persist_goal_desc")

        all_sources = ["arxiv", "pubmed", "biorxiv", "ieee_xplore", "scopus", "google_scholar", "semantic_scholar"]
        selected_sources = st.multiselect("Bases de données pertinentes",
                                        options=all_sources,
                                        key="persist_sources",
                                        on_change=lambda: save_config({"source_type": st.session_state.persist_sources}))

    constraints = st.text_area("Contraintes (une par ligne)",
                             key="persist_constraints")

    st.markdown("---")
    submit_btn = st.button("Lancer la Recherche", type="primary", use_container_width=True)


# --- LOGIQUE D'EXECUTION ---
async def run_research_cycle():
    cs = CoScientist(use_local_llm=use_local_llm, enable_rag=enable_rag)
    st.session_state.co_scientist = cs

    constraint_list = [c.strip() for c in constraints.split('\n') if c.strip()]
    await cs.initialize_research_goal(
        title=goal_title,
        description=goal_desc,
        domain=goal_domain,
        constraints=constraint_list
    )

    status_container = st.status("Démarrage du workflow...", expanded=True)

    try:
        status_container.write(f"📚 Agent Littérature : Recherche sur {','.join(selected_sources).upper()}...")
        papers = await cs.run_literature_search(max_results=max_papers, sources=selected_sources)
        if papers:
            status_container.write(f"✅ {len(papers)} papiers pertinents trouvés.")
        else:
            status_container.write("⚠️ Aucun papier trouvé ou erreur (vérifiez votre connexion/dépendances).")

        status_container.write("🔬 Agent Génération : Création des hypothèses (avec contexte)...")
        await cs.run_hypothesis_generation_cycle(num_hypotheses=num_hypotheses)
        status_container.write(f"✅ {num_hypotheses} hypothèses générées.")

        for i in range(num_iterations):
            status_container.update(label=f"Itération {i+1}/{num_iterations} en cours...", state="running")

            status_container.write(f"📝 Agent Réflexion : Revue critique (Cycle {i+1})...")
            await cs.run_review_cycle()

            status_container.write(f"🔗 Agent Proximité : Analyse des similarités (Cycle {i+1})...")
            await cs.proximity_agent.compute_proximity(list(cs.context_memory.hypotheses.values()))

            status_container.write(f"🏆 Agent Classement : Tournoi LLM-as-judge (Cycle {i+1})...")
            await cs.run_tournament_cycle(num_matches=num_hypotheses)

            status_container.write(f"🧬 Agent Evolution : Amélioration des idées (Cycle {i+1})...")
            await cs.run_evolution_cycle()

            status_container.write(f"🧪 Agent Expérimentation : Tests empiriques (Cycle {i+1})...")
            await cs.run_experiment_cycle()

            status_container.write(f"🎯 Agent Meta-Review : Synthèse (Cycle {i+1})...")
            await cs.run_meta_review_cycle()

            # Auto-save after each iteration
            try:
                saved_path = save_session(cs, goal_title)
                status_container.write(f"💾 Session sauvegardée: {os.path.basename(saved_path)}")
            except Exception as save_err:
                logger.warning("Auto-save failed: %s", save_err)

        status_container.update(label="Recherche terminée !", state="complete", expanded=False)
        st.session_state.results = cs

    except Exception as e:
        logger.error("Erreur critique dans run_research_cycle: %s", str(e), exc_info=True)
        status_container.update(label="Erreur critique", state="error")
        st.error(f"Une erreur fatale est survenue dans le workflow: {str(e)}")

    if cs.generation_agent.last_error:
        st.warning(f"⚠️ Note: Le générateur a rencontré une erreur et a utilisé la simulation :\n\n{cs.generation_agent.last_error}")


if submit_btn:
    save_config({
        "goal_title": goal_title,
        "goal_domain": goal_domain,
        "goal_desc": goal_desc,
        "constraints": constraints
    })
    st.session_state.is_running = True

    with st.expander("📝 Logs d'exécution détaillés", expanded=True):
        log_container = st.empty()

    redirector = StreamlitRedirector(log_container)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = redirector
    sys.stderr = redirector

    # Re-route python logging to the redirector specifically
    log_handler = logging.StreamHandler(redirector)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(log_handler)
    # Ensure root level is logging so we actually see INFO/WARNING
    logging.getLogger().setLevel(logging.INFO)

    try:
        # Safe asyncio execution via dedicated thread (avoids ProactorEventLoop conflicts on Windows)
        run_async(run_research_cycle())
    except Exception as e:
        st.error(f"Erreur fatale: {e}")
        logger.error("Fatal error in submit handler: %s", e, exc_info=True)
    finally:
        # Flush whatever is left
        redirector.flush()
        # Clean up handlers
        logging.getLogger().removeHandler(log_handler)
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    st.session_state.is_running = False
    st.rerun()


# --- SECTION 2: RESULTATS ---
if st.session_state.results:
    cs = st.session_state.results
    
    # Affichage des erreurs de génération persistantes
    if cs.generation_agent.last_error:
        st.warning(f"⚠️ **Attention : Le générateur a rencontré une erreur et a utilisé la simulation.**\n\n**Détail de l'erreur :**\n{cs.generation_agent.last_error}")
    
    hypotheses = list(cs.context_memory.hypotheses.values())
    
    # Métriques Globales
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hypothèses Totales", len(hypotheses))
    m2.metric("Matchs Tournoi", len(cs.context_memory.tournament_history))
    m3.metric("Revues Effectuées", cs.reflection_agent.reviews_completed)
    
    # Meilleure Hypothèse
    top_hyp = max(hypotheses, key=lambda h: h.elo_rating)
    m4.metric("Meilleur Elo", f"{top_hyp.elo_rating:.0f}")

    # --- ONGLETS D'ANALYSE ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏆 Classement", "📚 Littérature", "🕸️ Knowledge Graph", "📊 Métriques", "📝 Meta-Review", "💾 Export"])
    
    with tab1:
        st.subheader("Classement des Hypothèses")
        
        # Conversion en DataFrame pour affichage propre
        data = []
        for h in hypotheses:
            data.append({
                "ID": h.id,
                "Titre": h.title,
                "Elo": int(h.elo_rating),
                "Nouveauté": h.novelty_level,
                "Status": h.status.value,
                "Reviews": len(h.reviews)
            })
        df = pd.DataFrame(data).sort_values("Elo", ascending=False)
        
        # Affichage interactif avec Edition
        edited_df = st.data_editor(
            df,
            column_config={
                "Elo": st.column_config.ProgressColumn(
                    "Score Elo",
                    help="Classement relatif",
                    format="%d",
                    min_value=1000,
                    max_value=1600,
                ),
                "Nouveauté": st.column_config.TextColumn(
                    "Niveau de Nouveauté",
                ),
            },
            hide_index=True,
            width="stretch",
            disabled=["ID", "Elo", "Status", "Reviews", "Titre", "Nouveauté"] # Read-only for now to avoid state mismatch
        )
        
        st.subheader("Détails des Hypothèses")
        selected_id = st.selectbox("Choisir une hypothèse pour voir les détails", df["ID"].tolist(), format_func=lambda x: df[df["ID"]==x]["Titre"].values[0])
        
        if selected_id:
            h = cs.context_memory.hypotheses[selected_id]
            
            with st.container():
                st.markdown(f"""
                <div class="hypothesis-card">
                    <h3>{h.title}</h3>
                    <p style="color: #6c757d; font-style: italic; margin-bottom: 15px;">{h.description}</p>
                    <div style="background-color: rgba(66, 133, 244, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        <strong>🧠 Raisonnement & Formulation (Divergence Phase 3):</strong><br>
                        {h.reasoning if h.reasoning else "Généré via la méthode : " + h.generation_method}
                    </div>
                    <p><strong>⚙️ Mécanisme Scientifique:</strong><br>{h.mechanism}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if hasattr(h, 'experimental_results') and h.experimental_results:
                    st.markdown("#### 🧪 Résultats Expérimentaux Empiriques (Code Exécuté)")
                    st.code(h.experimental_results, language="log")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🧪 Prédictions Testables")
                    if h.testable_predictions:
                        for p in h.testable_predictions:
                            st.markdown(f"- {p}")
                    else:
                        st.info("Aucune prédiction générée.")
                
                with c2:
                    st.markdown("#### 📚 Preuves & Sources (Citation Tracking)")
                    # Affichage combiné des preuves et des papiers cités
                    if h.grounding_evidence:
                        for g in h.grounding_evidence:
                            st.markdown(f"- {g}")
                    else:
                        st.info("Aucune preuve spécifique générée.")
                        
                    if h.cited_papers:
                        st.markdown("**Références Scientifiques Strictes:**")
                        for p in h.cited_papers:
                            st.markdown(f"- *{p}*")
                
                if h.reviews:
                    st.markdown("#### 🧐 Dernières Critiques")
                    last_review = h.reviews[-1]
                    st.info(f"**Feedback:** {last_review.feedback}")
                    
                    # Scores sous forme de jauges
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.progress(last_review.correctness_score, text="Correctness")
                    sc2.progress(last_review.novelty_score, text="Novelty")
                    sc3.progress(last_review.testability_score, text="Testability")
                    sc4.progress(last_review.quality_score, text="Quality")

    with tab2:
        # Bug 2: Fix undefined source_type
        sources_display = ", ".join(selected_sources).upper() if 'selected_sources' in locals() else "SÉLECTIONNÉES"
        st.subheader(f"Contexte Bibliographique ({sources_display})")
        # Accès direct à la mémoire du contexte
        papers = cs.context_memory.literature_context
        if papers:
            for p in papers:
                st.markdown(f"""
                <div class="paper-card">
                    <h4><a href="{p['url']}" target="_blank" style="color: #60a5fa; text-decoration: none;">{p['title']}</a></h4>
                    <p style="font-size: 0.9em; color: #cbd5e1;">📅 {p['published']} | ✍️ {', '.join(p['authors'][:3])}...</p>
                    <p style="font-size: 0.95em;">{p['summary']}</p>
                    <p style="font-size: 0.8em; color: #94a3b8;">Source: {p.get('source', 'Unknown')} | RAG ID: {hashlib.md5(p.get('url', '').encode()).hexdigest()[:8] if hasattr(hashlib, 'md5') else 'N/A'}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucun papier trouvé ou recherche désactivée.")

    with tab3:
        st.subheader("🕸️ Graphe de Connaissances Extrait (GraphRAG)")
        st.markdown("Extraction automatique des relations inter-entités pour découvrir des mécanismes croisés.")
        if hasattr(cs, 'graph_agent') and hasattr(cs.graph_agent, 'graph') and cs.graph_agent.graph:
            graph_data = []
            for subj, relations in cs.graph_agent.graph.items():
                for rel in relations:
                    graph_data.append({"Sujet (Entité)": subj, "Relation -> Objet": rel})
            
            st.dataframe(pd.DataFrame(graph_data).head(100), width="stretch")
        else:
            st.info("Le graphe de connaissances n'est pas disponible pour cette session.")

    with tab4:
        st.subheader("Distribution des Scores Elo")
        fig = px.bar(df, x='Titre', y='Elo', color='Nouveauté', 
                     title="Classement Elo par Nouveauté",
                     color_discrete_map={'low': '#94a3b8', 'medium': '#60a5fa', 'high': '#3b82f6', 'very_high': '#8b5cf6'})
        st.plotly_chart(fig, width="stretch")
        
        st.subheader("Relation Qualité vs Nouveauté")
        # Préparer données pour scatter plot
        scatter_data = []
        for h in hypotheses:
            if h.reviews:
                last_r = h.reviews[-1]
                scatter_data.append({
                    "Titre": h.title,
                    "Quality": last_r.quality_score,
                    "Novelty Score": last_r.novelty_score,
                    "Elo": h.elo_rating
                })
        if scatter_data:
            df_scatter = pd.DataFrame(scatter_data)
            fig2 = px.scatter(df_scatter, x="Novelty Score", y="Quality", size="Elo", hover_name="Titre",
                              title="Qualité vs Nouveauté (Taille = Elo)")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("Pas assez de données de revue pour le graphique.")

    with tab5:
        st.subheader("Synthèse de Recherche (Meta-Review)")
        # Récupérer la dernière meta-review si disponible (c'est un dict retourné par la fonction, mais pas stocké directement dans context_memory de manière simple dans le code original, on va simuler un appel ou le récupérer si on l'avait stocké. 
        # Pour l'instant, on va régénérer une vue rapide ou afficher l'overview)
        
        # Note: Dans l'implémentation actuelle, meta_review est retourné mais pas persisté dans context_memory explicitement sauf via logs.
        # On va demander à l'agent de le refaire rapidement pour l'affichage
        if st.button("Générer le rapport final"):
            with st.spinner("Génération du rapport..."):
                try:
                    # Fix: Use new event loop to avoid conflict with streamlit loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    mr = loop.run_until_complete(cs.meta_review_agent.generate_meta_review(
                        list(cs.context_memory.hypotheses.values()),
                        cs.context_memory.tournament_history,
                        cs.context_memory.research_goal
                    ))
                    loop.close()
                    st.markdown(mr['research_overview'])
                    
                    st.markdown("### 💡 Suggestions d'amélioration")
                    for imp in mr['suggested_improvements']:
                        st.markdown(f"- {imp}")
                except RuntimeError as e:
                    st.error(f"Erreur d'exécution asynchrone : {e}. Essayez de relancer l'application.")
                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la génération : {e}")

    with tab6:
        st.subheader("Exporter les données")
        
        # Préparation du JSON
        json_str = json.dumps({
            "goal": asdict(cs.context_memory.research_goal),
            "literature_context": cs.context_memory.literature_context,
            "hypotheses": [asdict(h) for h in hypotheses]
        }, indent=2, default=str)
        
        st.download_button(
            label="📥 Télécharger le rapport brut JSON",
            data=json_str,
            file_name="co_scientist_results.json",
            mime="application/json"
        )
        
        st.divider()
        st.subheader("📄 Génération de l'Article Scientifique (Phase 6)")
        st.markdown("Rédigez un article formel PDF incluant les hypothèses, les graphes de connaissances (GraphRAG) et les réflexions générées par l'IA.")
        
        if st.button("Générer l'Article Scientifique Complet (PDF)", type="primary"):
            with st.spinner("Rédaction et compilation pdflatex en cours... (Patientez, cela nécessite plusieurs appels LLM)"):
                try:
                    import subprocess
                    result = subprocess.run([sys.executable, "scripts/generate_paper.py"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ Article généré avec succès ! (ai_co_scientist_paper_detailed.pdf)")
                        # Offer download
                        pdf_path = "ai_co_scientist_paper_detailed.pdf"
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                st.download_button("📥 Télécharger le PDF Final", f, file_name="ai_co_scientist_paper_detailed.pdf", mime="application/pdf")
                    else:
                        st.error(f"❌ Erreur de génération: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Impossible de lancer la génération: {e}")
