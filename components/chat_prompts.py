from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
import re

type_instructions_map = {
    "factoid": "Answer precisely and factually with info from abstracts only.",
    "summary": "Summarize the main findings coherently.",
    "yesno": "Answer with 'Yes' or 'No' and justify with citations.",
    "list": "Answer precisely and factually with info from abstracts only."
}

def get_type_instructions(qtype):
    if qtype is None:
        return "Answer precisely and factually with info from abstracts only."
    return type_instructions_map.get(qtype, "Answer precisely and factually with info from abstracts only.")


def format_documents_for_prompt(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", f"Document {i}")
        doi = doc.metadata.get("doi") or doc.metadata.get("source", "DOI not available")
        abstract = re.sub(r'\s+', ' ', doc.page_content.strip())
        formatted.append(f"[{i}]\nTitle: {title}\nAbstract: {abstract}")
    return "\n\n".join(formatted)


chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a knowledgeable expert chatbot in the biomedicine field."),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            """
            Answer the following scientific question: {question}, 
            using the following context retrieved from scientific articles: {retrieved_abstracts}.

            The user might refer to the history of your conversation. Please, use the following history of messages for the context as you see fit.

            The abstracts will come formatted in the following way: ABSTRACT TITLE: <abstract title>; ABSTRACT CONTENT: <abstract content>, ABSTRACT DOI: <abstract doi> (the content inside <> will be variable).
            In your answer, ALWAYS cite the abstract title and abstract DOI when citing a particular piece of information from that given abstract.

            Your example response might look like this:

            In the article (here in the brackets goes the contents of ABSTRACT_TITLE), it was discussed, that Cannabis hyperemesis syndrome (CHS) is associated with chronic, heavy cannabis use. The endocannabinoid system (ECS) plays a crucial role in the effects of cannabis on end organs and is central to the pathophysiology of CHS. (here, in the end of the cited chunk, the ABSTRACT_DOI goes)
            """
        ),
    ]
)

qa_template = PromptTemplate(
    input_variables=['question', 'retrieved_abstracts', 'type_instructions'],
    template="""
        Answer the following scientific question: {question}, 
        using the following context retrieved from scientific articles: {retrieved_abstracts}.

        The abstracts will come formatted in the following way: ABSTRACT TITLE: <abstract title>; ABSTRACT CONTENT: <abstract content>, ABSTRACT DOI: <abstract doi> (the content inside <> will be variable).
        In your answer, ALWAYS cite the abstract title and abstract DOI when citing a particular piece of information from that given abstract.

        Type-specific instructions:
        {type_instructions}

        Your example response might look like this:

        In the article (here in the brackets goes the contents of ABSTRACT_TITLE), it was discussed, that Cannabis hyperemesis syndrome (CHS) is associated with chronic, heavy cannabis use. The endocannabinoid system (ECS) plays a crucial role in the effects of cannabis on end organs and is central to the pathophysiology of CHS. (here, in the end of the cited chunk, the ABSTRACT_DOI goes)
    """
)