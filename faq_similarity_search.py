from ollama import chat, ChatResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

client = QdrantClient("http://localhost:6333")

faq_ctic = [
    {
        "pergunta": "O CTIC faz manutenção em impressoras?",
        "resposta": "Não, apenas realiza instalação e configuração de impressoras."
    },
    {
        "pergunta": "Posso levar meu computador pessoal ao CTIC para manutenção?",
        "resposta": "Não, os serviços de manutenção são exclusivamente para computadores que pertencem à UFPA. Os computadores que chegam ao CTIC pertencem às unidades acadêmicas e administrativas da Universidade e só são recebidos com autorização dos técnicos alocados no atendimento do SAGITTA."
    },
    {
        "pergunta": "O CTIC faz manutenção em Smartphones e tablets?",
        "resposta": "Não, apenas disponibiliza um tutorial para configuração da rede sem fio institucional com o nome 'UFPA 2.0 - Institucional'."
    },
    {
        "pergunta": "O CTIC disponibiliza ou configura alguma rede sem fio não homologada pela solução aplicada na UFPA?",
        "resposta": "Não, as únicas redes homologadas, disponibilizadas e gerenciadas pelo CTIC são 'UFPA 2.0 - Institucional' e 'Eduroam', qualquer outra rede com nome diferente não é mantida e administrada pela equipe do CTIC."
    },
    {
        "pergunta": "O CTIC utiliza rádios wireless convencionais ou soluções caseiras?",
        "resposta": "Não, os rádios homologados na rede sem fio mantida pelo CTIC são partes de uma solução corporativa implementada na UFPA, pois possuem uma capacidade de gerência, processamento e transmissão de dados maior que rádios convencionais ou caseiros, o que facilita a administração."
    },
    {
        "pergunta": "O CTIC realiza backup de dados dos sites hospedados em seu data center?",
        "resposta": "Não, por motivos de espaço em seu storage de armazenamento, apenas os serviços críticos da Universidade (E-mail, SIG-UFPA, Portal, etc.) possuem uma rotina de backups, o backup dos arquivos dos sites dos usuários é de responsabilidade do mesmo conforme está descrito no termo de compromisso assinado no momento da solicitação de hospedagem do site."
    },
    {
        "pergunta": "O CTIC envia E-mail solicitação atualização cadastral, dados pessoais e dados acadêmicos?",
        "resposta": "Não, nenhuma coordenadoria envia e-mail para os usuários solicitando tais informações, caso o usuário precise alterar algum dado e não conseguir via sistema deve abrir uma solicitação no serviço de atendimento do CTIC, Sagitta, e aguardar o atendimento."
    },
    {
        "pergunta": "O CTIC realiza algum bloqueio de páginas de internet?",
        "resposta": "Não, no momento nenhuma página de conteúdo web é bloqueada pelo CTIC."
    },
    {
        "pergunta": "O CTIC monitora e registra os acessos de usuários e dispositivos de todo o tráfego de dados na rede da UFPA?",
        "resposta": "Sim, as equipes de Data Center e Redes monitoram todo o tráfego de dados no ambiente computacional da UFPA, todo o processo de monitoramento está de acordo com a Lei Nº 12.965 intitulada 'Marco Civil da Internet', que determina responsabilidades ao CTIC como provedor de acesso à internet da Comunidade Universitária."
    },
    {
        "pergunta": "O CTIC realiza serviços de cabeamento estruturado e óptico nos prédios da UFPA?",
        "resposta": "Não, a equipe técnica do CTIC apenas fiscaliza obras de cabeamento estruturado e instalação de fibra óptica, o lançamento de cabos e conectorização é realizado por empresa especializada com profissionais treinados para a tarefa."
    }
]

historico_mensagens = []

def deletar_colecao(client, collection_name):
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Coleção '{collection_name}' deletada com sucesso.")
    except Exception as e:
        print(f"Erro ao deletar a coleção '{collection_name}': {e}")


def armazenar_embeddings():
    embeddings = []
    for item in faq_ctic:
        text = item['pergunta'] + " " + item['resposta']
        embedding = model.encode(text, convert_to_tensor=True).to(device)
        embeddings.append(embedding)

    collection_name = "faq_ctic"

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 384, "distance": "Cosine"}
        )
        print(f"Coleção '{collection_name}' criada com sucesso.")
    except Exception as e:
        print(f"Coleção '{collection_name}' já existe. Ignorando criação. Erro: {e}")

    for i, embedding in enumerate(embeddings):
        existing_point = client.search(
            collection_name=collection_name,
            query_vector=embedding.tolist(),
            limit=1
        )

        if not existing_point:
            client.upsert(
                collection_name=collection_name,
                points=[
                    {"id": i, "vector": embedding.tolist(), "payload": faq_ctic[i]}
                ]
            )

    print("Embeddings armazenados com sucesso no Qdrant!")



def buscar_documentos_relevantes(query, top_k=3, similaridade_minima=0.5):
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    results = client.search(
        collection_name="faq_ctic",
        query_vector=query_embedding.tolist(),
        limit=top_k
    )

    documentos_relevantes = [result.payload for result in results if result.score >= similaridade_minima]

    return documentos_relevantes



def consultar_modelo_local(consulta_usuario, documentos_relevantes, historico_mensagens):
    dados_faq = "\n".join([doc['resposta'] for doc in documentos_relevantes])
    full_prompt = f"Dados do FAQ:\n{dados_faq}\nPergunta do usuário: {consulta_usuario}"

    historico_mensagens.append({"role": "user", "content": consulta_usuario})

    response: ChatResponse = chat(model="llama3.2", messages=historico_mensagens + [
        {"role": "system", "content": "Você é um assistente útil que responde perguntas com base em FAQs fornecidas."},
        {"role": "user", "content": full_prompt}
    ], options={"temperature": 0.6})

    resposta_content = response.message.content

    historico_mensagens.append({"role": "assistant", "content": resposta_content})

    return resposta_content



def executar_fluxo(consulta_usuario, top_k=3, historico_mensagens=[]):
    documentos_relevantes = buscar_documentos_relevantes(consulta_usuario, top_k)
    resposta_final = consultar_modelo_local(consulta_usuario, documentos_relevantes, historico_mensagens)
    return resposta_final


def inicializar():
    armazenar_embeddings()


if __name__ == "__main__":
    inicializar()
    # deletar_colecao(client, "faq_ctic")

    print("Bem-vindo ao sistema de FAQ. Faça sua pergunta ou digite 'q' para sair.")

    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() == "q":
            print("Encerrando o sistema de FAQ. Até logo!")
            break

        resposta_final = executar_fluxo(pergunta, historico_mensagens=historico_mensagens)

        print(f"\nCTIC: {resposta_final}")
