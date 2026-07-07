import streamlit as st
from config.prompt import get_rag_chain


st.set_page_config(page_title="Agente Jurídico RAG", page_icon="⚖️", layout="centered")

st.title("⚖️ Agente Jurídico RAG")
st.markdown("Protótipo de avaliação de resistência contra Prompt Injection e RAG Poisoning.")

st.sidebar.header("🛡️ Configurações de Segurança")
defense_mode = st.sidebar.radio(
    "Selecione o Modo do Agente:",
    ("baseline", "C"),
    format_func=lambda x: "🔴 Baseline (Desprotegido)" if x == "baseline" else "🟢 Configuração C (SOTA)"
)

if "current_mode" not in st.session_state or st.session_state.current_mode != defense_mode:
    st.session_state.current_mode = defense_mode
    st.session_state.messages = []
    with st.spinner("Carregando modelo e banco vetorial na GPU..."):
        st.session_state.chain = get_rag_chain(defense_mode=defense_mode)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Faça uma pergunta legítima ou tente um ataque adversarial..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recuperando documentos e analisando segurança..."):
            try:
                response = st.session_state.chain.invoke(prompt)
                
                if "SECURITY ALERT" in response or "STRUCTURAL ALERT" in response:
                    st.error(response)
                else:
                    st.success(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Ocorreu um erro no processamento: {e}")