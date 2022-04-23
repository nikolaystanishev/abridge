import React, {useEffect} from "react";
import {AnalysisT, DataObjectT, LabelT, PlatformsFilterT} from "../../types/types";
import axios from "axios";
import {Container, Grid} from "@mui/material";
import {TwitterTweetEmbed} from "react-twitter-embed";
import PieChart, {Connector, Label, Series, Size,} from 'devextreme-react/pie-chart';
import {Font} from "devextreme-react/chart";


export function Analysis(props: { filter: PlatformsFilterT[], reload: boolean }) {
  const [analysis, setAnalysis] = React.useState<AnalysisT>();

  useEffect(() => {
    if (props.reload) {
      axios.post('api/analysis', props.filter).then(response => {
        setAnalysis(response.data);
      });
    }
  }, [props.reload]);

  return (
    <>
      {analysis && (
        <Grid container spacing={3} width="0.8" margin="auto" border={1} borderColor="primary.main" borderRadius={2}>
          <Grid item xs>
            {analysis.data.map((item: DataObjectT) => {
              return (
                <Container key={item.id} maxWidth="sm"
                           style={{
                             // @ts-ignore
                             backgroundColor: item.label == LabelT.POSITIVE ? '#4caf50' : '#ef5350',
                             paddingTop: '14px',
                             paddingBottom: '14px',
                             marginTop: '10px',
                             marginBottom: '10px',
                             borderRadius: '12px'
                           }}>
                  <TwitterTweetEmbed tweetId={item.id}/>
                </Container>
              );
            })}
          </Grid>
          <Grid item xs>
            <PieChart
              dataSource={Object.values(analysis.data.map((item: DataObjectT) => item.label == LabelT.POSITIVE ? 'Positive' : 'Negative').reduce((data: { [key: string]: { label: string, value: number } }, label: string) => {
                data[label].value += 1;
                return data;
              }, {'Positive': {label: 'Positive', value: 0}, 'Negative': {label: 'Negative', value: 0}}))}
              title="Distribution of Public Opinion"
              customizePoint={(point: any) => {
                return {
                  color: point.argument == 'Positive' ? '#4caf50' : '#ef5350',
                };
              }}
            >
              <Series
                argumentField="label"
                valueField="value">
                <Label
                  visible={true}
                  position="columns"
                  customizeText={(arg: any) => `${arg.valueText} (${arg.percentText})`}>
                  <Font size={16}/>
                  <Connector visible={true} width={0.5}/>
                </Label>
              </Series>

              <Size width={500}/>
            </PieChart>
          </Grid>
        </Grid>
      )}
    </>
  )
}