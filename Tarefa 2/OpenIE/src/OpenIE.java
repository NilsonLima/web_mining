import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.simple.*;

public class OpenIE {

	public static void main(String[] args){
		Document doc = new Document("Obama was born in Hawaii. He is our president.");
		
		for(Sentence sent: doc.sentences()){
			for(RelationTriple triple: sent.openieTriples()){
				
				System.out.println(triple.confidence + "\t" +
			            triple.subjectLemmaGloss() + "\t" +
			            triple.relationLemmaGloss() + "\t" +
			            triple.objectLemmaGloss());
			}
		}
	}
}
