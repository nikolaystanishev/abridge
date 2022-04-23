import React, {useEffect} from "react";
import {AnalysisT, DataObjectT, LabelT, PlatformsFilterT} from "../../types/types";
import axios from "axios";
import {Container, Grid} from "@mui/material";
import {TwitterTweetEmbed} from "react-twitter-embed";

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
      <Grid container spacing={3} width="0.8" margin="auto" border={1} borderColor="primary.main" borderRadius={2}
            style={analysis != undefined ? {} : {display: 'none'}}>
        <Grid item xs>
          {analysis?.data.map((item: DataObjectT) => {
            return (
              <Container key={item.id} maxWidth="sm"
                         style={{
                           // @ts-ignore
                           backgroundColor: LabelT[item.label] == LabelT.POSITIVE ? '#4caf50' : '#ef5350',
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
          
        </Grid>
      </Grid>
    </>
  )
}