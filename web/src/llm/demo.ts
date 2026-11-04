import type { ChatMessage, CompleteOptions, CompletionResult, Engine, ToolCall } from './types';

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const FORMULA_EXPLANATION = `The profitability formula is decomposed into the following P&L terms, each with a clear business meaning:

| Term | Formula | Sign | Business Meaning |
| :--- | :--- | :--- | :--- |
| **rev_discount** | 0.0240 × S_all | + | Merchant discount (Interchange) revenue on total billed business (S_all = f6+f7+f8+f9+f10) |
| **rev_interest** | 0.331 × f1 | + | Net Interest Income on average revolving balance (f1) |
| **cost_lounge** | −55 × f13 | − | Lounge access cost at fully-loaded issuer cost ($55 per visit) |
| **cost_cab** | −29 × f15 | − | Cab credit cost at $29 per month utilized |
| **cost_airline_credit** | −1.9 × f14 | − | Airline fee credit cost at 1.9× loaded multiplier |
| **cost_entertainment_credit** | −1.5 × f16 | − | Entertainment credit cost at 1.5× loaded multiplier |
| **cost_credit_loss** | −1.35 × f11 × f1 | − | Expected write-offs: risk rate (f11) × revolving exposure (f1) |
| **cost_cancel_calls** | −70 × f2 | − | Retention cost per cancellation call (f2) |
| **cost_collections** | −8500 × f3 | − | Collections-driven cancellation cost — acts as a soft exclusion; business logic: non-retainable |
| **cost_servicing** | −4.5 × f12 | − | Digital servicing cost per website login (f12) |

All missing values are scored as 0 ("no recorded activity"). The final profit is the sum of all terms, ranked descending, with the top 20% defined as those with percentile rank ≥ 0.8. This structure was derived from single-lever response-curve analysis and validated via deletion/falsification tests.`;

export class DemoEngine implements Engine {
  readonly kind = 'demo';
  readonly label = 'Demo LLM (Vercel Hostable)';
  readonly contextTokens = 8192;

  async complete(messages: ChatMessage[], opts?: CompleteOptions): Promise<CompletionResult> {
    const last = messages[messages.length - 1];

    // Round 2: Tool execution finished, return narrative
    if (last && last.role === 'tool') {
      let narrative = '';
      try {
        const res = JSON.parse(last.content);
        if (last.tool_name === 'get_customer_profile') {
          if (res.customer_id === 1) {
            narrative =
              'Customer 1 scores **-683.52**, well below the top-20% cutoff of $2,012.86 (percentile rank ~15.2%). Revenue is modest — $355.52 discount revenue on $13,865.40 in billed business and $0 net interest (no revolving balance) — while benefit and servicing costs total $1,038.40, led by the airline credit ($-372.40) and cab credit ($-145.00). No attrition flags; this is a light-revenue, benefit-heavy transactor with no revolving activity.';
          } else if (res.customer_id === 20) {
            narrative =
              'Customer 20 ranks in the top 20% (percentile rank ~96.4%, rank 18,182 of 500,000) with a profit of **$5,950.21**, well above the $2,012.86 cutoff.\n\nProfitability is overwhelmingly driven by **Net Interest Income** ($5,947.32) on an average revolving balance of $17,967.73 (f1), alongside **Discount Revenue** ($882.11) on $36,754.77 total spend. These gains far outweigh credit loss costs ($-442.62) and benefit usage (lounge: $-165.00, entertainment: $-96.61, digital servicing: $-117.00).';
          } else if (res.error) {
            narrative = `Error retrieving customer profile: ${res.error}. ${res.hint ?? ''}`;
          } else {
            const profitStr = `${res.profit < 0 ? '-' : ''}$${Math.abs(res.profit).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            const terms = res.terms || {};
            const costTotal = Math.abs(
              (terms.cost_lounge || 0) +
                (terms.cost_cab || 0) +
                (terms.cost_airline_credit || 0) +
                (terms.cost_ent_credit || 0) +
                (terms.cost_servicing || 0),
            );
            narrative = `Customer ${res.customer_id} scores **${profitStr}**, ${res.is_top20 ? 'ranking firmly in the top 20%' : 'falling below the top-20% cutoff of $2,012.86'} (percentile rank ~${((res.percentile_rank || 0) * 100).toFixed(1)}%, rank ${(res.rank_out_of_500k || 0).toLocaleString()} of 500,000). Total discount revenue on billed business is $${(terms.rev_discount || 0).toFixed(2)}, net interest income is $${(terms.rev_interest || 0).toFixed(2)}, and operational benefit & servicing costs total $${costTotal.toFixed(2)}.`;
          }
        } else if (last.tool_name === 'get_distribution') {
          narrative =
            'Across all 500,000 accounts in the dataset, profitability displays a pronounced right tail. The top 20% cutoff threshold is **$2,012.86**, with top-quintile customers generating an average profit of **$4,354.85**. In contrast, the remaining 80% averages **-$922.40**, depressed primarily by collections-driven exclusions and accounts where card benefit utilization exceeds billed spend interchange.';
        } else if (last.tool_name === 'segment_compare') {
          narrative =
            'Comparing **Lounge Access Count** (`f13`) between segments:\n- **Top 20% Mean:** 0.58 visits/year\n- **Bottom 80% Mean:** 0.46 visits/year\n\nWhile top-20% cardholders access lounges slightly more often on average, median visits for both segments are 0 (75th percentile is 1.0). At $55/visit, lounge expenses contribute only a minor cost drag compared to interchange revenue and revolving interest income.';
        } else if (last.tool_name === 'term_contribution_summary') {
          narrative =
            'Looking across the ten P&L components, **collections cancellations** (`cost_collections`) hurts profitability the most by far for the general portfolio, causing an average drag of **-$1,153.96** across non-top-20% customers (and $0 in the top 20% due to strict non-retainable exclusion logic).\n\nAmong operational costs, **expected credit loss** (`cost_credit_loss`) is the next largest drag (-$175.20 in top 20%, -$226.48 in rest), followed by **cab benefits** (-$132.27 in top 20% vs -$107.70 in rest), **digital servicing** (-$120.25 vs -$135.02), and **airline credits** (-$98.35 vs -$74.88).';
        } else {
          narrative = `Tool execution complete. Result summary: ${JSON.stringify(res).slice(0, 300)}`;
        }
      } catch {
        narrative = `Result: ${last.content.slice(0, 300)}`;
      }

      await this.stream(narrative, opts);
      return { content: narrative, toolCalls: [] };
    }

    // Round 1: Parse user intent from prompt
    const userPrompt = (last?.content || '').toLowerCase();

    // Check for customer breakdown intent
    const custMatch = userPrompt.match(/customer\s*#?\s*(\d+)/i) || userPrompt.match(/(\d+)/);
    if ((userPrompt.includes('customer') || userPrompt.includes('breakdown')) && custMatch) {
      const cid = parseInt(custMatch[1], 10);
      const toolCalls: ToolCall[] = [{ name: 'get_customer_profile', arguments: { customer_id: cid } }];
      return { content: '', toolCalls };
    }

    // Check for formula explanation
    if (userPrompt.includes('formula') || userPrompt.includes('term by term') || userPrompt.includes('explain my profit')) {
      await this.stream(FORMULA_EXPLANATION, opts);
      return { content: FORMULA_EXPLANATION, toolCalls: [] };
    }

    // Check for lounge comparison
    if (userPrompt.includes('lounge')) {
      const toolCalls: ToolCall[] = [{ name: 'segment_compare', arguments: { feature: 'f13' } }];
      return { content: '', toolCalls };
    }

    // Check for profit distribution
    if (userPrompt.includes('distribution') || userPrompt.includes('histogram')) {
      const toolCalls: ToolCall[] = [{ name: 'get_distribution', arguments: { column: 'profit' } }];
      return { content: '', toolCalls };
    }

    // Check for cost term hurts profitability
    if (userPrompt.includes('cost') || userPrompt.includes('hurts') || userPrompt.includes('drag')) {
      const toolCalls: ToolCall[] = [{ name: 'term_contribution_summary', arguments: {} }];
      return { content: '', toolCalls };
    }

    // General answers
    let text = '';
    if (userPrompt.includes('dataset') || userPrompt.includes('500,000') || userPrompt.includes('cutoff')) {
      text =
        'The ProfitLens dataset comprises **500,000 customers** evaluated on 23 financial and behavioral features (`f1` through `f23`). The top 20% cutoff is **$2,012.86**, with top-quintile accounts averaging **$4,354.85** in profit. Feel free to ask for any customer breakdown (e.g., "Profit breakdown for customer 1") or explore the P&L formula!';
    } else {
      text =
        'I am ProfitLens, the Premier Card profitability assistant. I can compute customer profitability breakdowns, analyze cost terms, compare segments, and explain the scoring formula. Try asking one of the sample questions on the left or asking about any customer (e.g., "Profit breakdown for customer 1")!';
    }

    await this.stream(text, opts);
    return { content: text, toolCalls: [] };
  }

  private async stream(text: string, opts?: CompleteOptions) {
    if (!opts?.onToken) return;
    const words = text.split(' ');
    let current = '';
    for (let i = 0; i < words.length; i++) {
      current += (i === 0 ? '' : ' ') + words[i];
      opts.onToken(current);
      if (opts.signal?.aborted) break;
      await delay(8);
    }
  }
}
