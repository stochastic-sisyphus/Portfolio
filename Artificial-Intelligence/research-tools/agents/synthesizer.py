from utils.nvidia_api import summarize_text, generate_questions, answer_question

class Synthesizer:
    def synthesize(self, topic, data):
        # Summarize the data
        summary = summarize_text(data)
        
        # Generate questions based on the summary
        questions_text = generate_questions(summary)
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and not q.startswith("Here are")]
        
        # Answer the generated questions using the full data
        qa_pairs = []
        for q in questions:
            if q.startswith(('Q:', 'Question:')):
                q = q.split(':', 1)[1].strip()
            answer = answer_question(q, data)
            qa_pairs.append((q, answer))
        
        # Combine the summary and Q&A into a final synthesis
        synthesis = f"Summary: {summary}\n\n"
        for i, (q, a) in enumerate(qa_pairs, 1):
            synthesis += f"Q{i}: {q}\nA{i}: {a}\n\n"
        
        return synthesis