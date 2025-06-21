import { glob } from 'astro/loaders';
import { defineCollection, z } from 'astro:content';


const notes = defineCollection({
	// Load Markdown and MDX files in the `src/content/notes/` directory.
	loader: glob({ base: './src/content/notes', pattern: '**/*.{md,mdx}' }),
	// Type-check frontmatter using a schema
	schema: ({ image }) => z.object({
		title: z.string(),
		uses_katex: z.boolean().optional(),
		// pubDate: z.coerce.date(),
	}),
});

export const collections = { notes };
