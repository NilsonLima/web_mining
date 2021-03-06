import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.simple.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import java.io.IOException;
import java.io.FileNotFoundException;

import java.util.regex.*;


public class InformationExtraction {
	
	private static BufferedReader br;
	private static BufferedWriter bw;
	private static String match;
	
	public static void main(String[] args) throws IOException{
		
		for(int i = 1; i <= 10; i++){
			
			try {
				br = new BufferedReader(new FileReader("/home/nilson/git/web_mining/Tarefa2/docbase/file" + i));
				bw = new BufferedWriter(new FileWriter("/home/nilson/git/web_mining/Tarefa2/extracted/file" + i));

				wrapper();
				openIE();
				
				br.close();
				bw.close();
				
			} 
			catch (FileNotFoundException e) {
				System.out.println("docbase/file" + i + " not found");
			}
		}
	}
	
	public static void wrapper() throws IOException{
		
		String link = br.readLine();
		String title = br.readLine();
		String authors = br.readLine();		
		
		match = br.readLine() + "\n" + br.readLine();
		
		String comments = extractPattern("Comments", match);
		String subjects = extractPattern("Subjects", match);
		
		bw.write("Authors: " + authors + "\n");
		bw.write("Title: " + title + "\n");
		bw.write("Subjects: " + subjects + "\n");
		bw.write("Comments: " + comments + "\n");
		bw.write("Link: " + "http://arxiv.org/abs/" + link.split(":")[1] + "\n\n");
		
	}
	
	public static void openIE() throws IOException{
		
		String line = br.readLine();
		
		if(line == null){
			line = match.split("\n")[1];
		}
		
		Document doc = new Document(line);
		
		for(Sentence sent: doc.sentences()){
			
			for(RelationTriple triple: sent.openieTriples()){
				
				bw.write(triple.relationGloss() + "("
			            						+ triple.subjectGloss() + ", "
			            	 					+ triple.objectGloss() + ")\n");
			}
		}
	}
	
	public static String extractPattern(String pattern, String match){
		
		Pattern p = Pattern.compile(pattern + ":[^\\s](.*[^\\n])");
		Matcher m = p.matcher(match);
		
		String ret = "null";
		
		if(m.find()){
			ret = m.group(1);
		}
		
		return ret;
	}
}
