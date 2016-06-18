import java.io.*;
import java.util.regex.*;
import java.util.Scanner;

public class ParseBooks {
	public static void main(String[] args)
	{
		StringBuilder wholeTextStringBuilder = new StringBuilder("");
		Scanner scanner = new Scanner(System.in);
		String line = "";
		do {
			line = scanner.nextLine() + " ";
			if (line.contains("[Illustration]"))
				continue;
			line = line.toLowerCase();
			wholeTextStringBuilder.append(line);
		} while (!line.contains("gutenberg") && scanner.hasNextLine());
		String wholeText = wholeTextStringBuilder.toString();
		wholeText = wholeText.replace("Mr.", "Mr");
		wholeText = wholeText.replace("Mrs.", "Mrs");
		String regex = "[^.!?]*[.!?]['\"]?";
		Matcher m = Pattern.compile(regex).matcher(wholeText);
		int count = 0;
		while (m.find()) {
			count++;
			System.out.println(wholeText.substring(m.start(), m.end()).trim());
		}
		


	}
}