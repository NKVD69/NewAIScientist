import { useState } from 'react';
import { Rocket, CheckCircle2, Play } from 'lucide-react';
import { workflowApi } from '../services/api';
import { PageHeader, Panel } from '../components/primitives/Panel';
import { Field, Input, Select, Textarea } from '../components/primitives/Field';
import Button from '../components/primitives/Button';
import { ErrorBanner } from '../components/primitives/Banner';

export default function Dashboard() {
  const [goal, setGoal] = useState({ title: '', description: '', domain: 'Biomedicine' });
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function initialize() {
    setLoading(true);
    setError(null);
    try {
      await workflowApi.initializeGoal(goal);
      setInitialized(true);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Research goal"
        lede="Everything downstream is anchored to this. The literature search, the hypotheses and the pre-registered predictions all inherit its framing, so be specific about the gap you want closed."
      />

      <ErrorBanner error={error} />

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Panel label="Mission" title="Define the question">
            <div className="space-y-5">
              <Field label="Title">
                <Input
                  value={goal.title}
                  onChange={(e) => setGoal({ ...goal, title: e.target.value })}
                  placeholder="Inhibition of the TLR4 pathway in neuroinflammation"
                />
              </Field>

              <Field
                label="Description and scope"
                hint="Name the specific gap. Vague goals produce hypotheses that are hard to falsify, and unfalsifiable predictions are excluded from adjudication."
              >
                <Textarea
                  rows={5}
                  value={goal.description}
                  onChange={(e) => setGoal({ ...goal, description: e.target.value })}
                  placeholder="Which mechanism is unresolved, and what would count as evidence either way?"
                />
              </Field>

              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Domain">
                  <Select
                    value={goal.domain}
                    onChange={(e) => setGoal({ ...goal, domain: e.target.value })}
                  >
                    <option>Biomedicine</option>
                    <option>Physics</option>
                    <option>Computer Science</option>
                    <option>Chemistry</option>
                  </Select>
                </Field>
                <Field label="Engine">
                  <div className="num border border-[var(--rule-strong)] bg-[var(--sunken)] px-3.5 py-2.5 text-[13px] text-[var(--ink-2)]">
                    configured server-side
                  </div>
                </Field>
              </div>

              <Button
                onClick={initialize}
                loading={loading}
                disabled={!goal.title.trim()}
                icon={initialized ? <CheckCircle2 className="h-4 w-4" /> : <Rocket className="h-4 w-4" />}
                className="w-full"
              >
                {initialized ? 'Goal set' : 'Set research goal'}
              </Button>
            </div>
          </Panel>
        </div>

        <div className="space-y-6">
          <Panel label="Autonomous" title="Full cycle">
            <p className="text-[13px] leading-relaxed text-[var(--ink-2)]">
              Runs every phase through to the manuscript. Expect a few hundred LLM calls; set a
              budget ceiling first if you are working against a paid endpoint.
            </p>
            <Button
              variant="secondary"
              className="mt-5 w-full"
              disabled={!initialized}
              icon={<Play className="h-4 w-4" />}
            >
              Start full cycle
            </Button>
            {!initialized && (
              <p className="mt-2.5 text-[11.5px] text-[var(--ink-3)]">
                Set a research goal first.
              </p>
            )}
          </Panel>

          {/*
            The previous dashboard showed "Total tokens 145k" and "Research
            confidence 87%" as hard-coded constants. In a product whose whole
            argument is honest uncertainty, a fabricated confidence figure is
            worse than no figure. Real meters live in the run rail, fed by the
            backend; nothing is invented here.
          */}
          <Panel label="Session" title="Meters">
            <p className="text-[13px] leading-relaxed text-[var(--ink-2)]">
              Budget, judge consistency and sandbox status are shown in the rail on the left,
              from the live session. They appear once a run starts.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  );
}
